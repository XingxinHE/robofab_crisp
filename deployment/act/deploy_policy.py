"""Deploy a pretrained policy with robofab-specific homing controls."""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import crisp_gym  # noqa: F401
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.policy import make_policy
from crisp_gym.policy.policy import list_policy_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging

from teleoperations.gamepad.home_config import get_gamepad_home_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a pretrained policy and record data in LeRobot format"
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--robot-type", type=str, default="franka")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--recording-manager-type", type=str, default="keyboard")
    parser.add_argument("--joint-control", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--path",
        "--model-path",
        dest="path",
        type=str,
        default=None,
        help="Path to pretrained_model. If omitted, prompts from outputs/train.",
    )
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--policy-config", type=str, default=None)
    parser.add_argument("--env-namespace", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument(
        "--home-config",
        type=str,
        default=None,
        help=(
            "Optional robot YAML/home config override. Accepts names such as "
            "'fr3_root_home_lab', 'robots/fr3_root_home_lab.yaml', "
            "'homes/table_a.yaml', or a file path."
        ),
    )
    parser.add_argument("--home-config-noise", type=float, default=0.0)
    parser.add_argument(
        "--after-teleop",
        type=str,
        default=None,
        help=(
            "Optional robot YAML/home config for final homing after deployment. "
            "If omitted, falls back to --home-config."
        ),
    )
    return parser.parse_args()


def select_model_path(path: str | None, logger: logging.Logger) -> str:
    if path is not None:
        return path

    logger.info("No path provided. Searching for models in 'outputs/train'.")
    models_path = Path("outputs/train")
    if not models_path.exists() or not models_path.is_dir():
        raise FileNotFoundError(
            "'outputs/train' does not exist. Provide a model path with --path."
        )

    models = sorted(str(model) for model in models_path.glob("**/pretrained_model"))
    if not models:
        raise FileNotFoundError(
            "No pretrained_model directories found under outputs/train. "
            "Provide a model path with --path."
        )
    return prompt.prompt(
        message="Please select a model to use for deployment:",
        options=models,
        default=models[0],
    )


def resolve_prompted_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset:",
        )
        logger.info("Using repository ID: %s", args.repo_id)

    args.path = select_model_path(args.path, logger)
    logger.info("Using model path: %s", args.path)

    if args.env_namespace is None:
        args.env_namespace = prompt.prompt(
            "Please enter the follower robot namespace:",
            default="right",
        )
        logger.info("Using follower namespace: %s", args.env_namespace)

    if args.env_config is None:
        follower_configs = list_env_configs()
        args.env_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info("Using follower configuration: %s", args.env_config)

    if args.policy_config is None:
        policy_configs = list_policy_configs()
        args.policy_config = prompt.prompt(
            "Please select the policy configuration to use.",
            options=policy_configs,
            default=policy_configs[0],
        )


def evaluation_output_file(args: argparse.Namespace) -> str:
    if not args.evaluate:
        return "evaluation_results.csv"

    datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        prompt.prompt(
            "Please enter the output file for evaluation results",
            default=f"evaluation_results_{args.path.replace('/', '_')}_{datetime_now}",
        )
        + ".csv"
    )


def home_for_deployment(
    env: ManipulatorBaseEnv,
    home_config: str | None,
    home_config_noise: float,
) -> list[float]:
    return get_gamepad_home_config(env, home_config, home_config_noise)


def home_after_deployment(
    env: ManipulatorBaseEnv,
    home_config: str | None,
    after_teleop: str | None,
    home_config_noise: float,
) -> list[float]:
    source = after_teleop if after_teleop is not None else home_config
    return get_gamepad_home_config(
        env,
        source,
        home_config_noise,
        config_key="after_teleop" if after_teleop is not None else None,
    )


def main() -> int:
    args = parse_args()
    logger = logging.getLogger(__name__)
    setup_logging(level=args.log_level)

    logger.info("-" * 40)
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info("  %-30s: %s", arg, value)
    logger.info("-" * 40)

    policy = None
    env = None
    try:
        resolve_prompted_args(args, logger)
        evaluation_file = evaluation_output_file(args)

        ctrl_type = "cartesian" if not args.joint_control else "joint"
        env = make_env(
            args.env_config, control_type=ctrl_type, namespace=args.env_namespace
        )

        features = get_features(env)
        evaluator = Evaluator(output_file="eval/" + evaluation_file)
        recording_manager = make_recording_manager(
            recording_manager_type=args.recording_manager_type,
            features=features,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            num_episodes=args.num_episodes,
            fps=args.fps,
            resume=args.resume,
            push_to_hub=args.push_to_hub,
        )
        recording_manager.wait_until_ready()

        logger.info("Setting up the policy.")
        policy = make_policy(
            name_or_config_name=args.policy_config,
            pretrained_path=args.path,
            env=env,
        )

        logger.info("Homing robot before starting deployment recording.")
        deployment_home = home_for_deployment(
            env, args.home_config, args.home_config_noise
        )
        env.home(home_config=deployment_home)
        env.reset()

        def on_start() -> None:
            env.reset()
            policy.reset()
            evaluator.start_timer()

        def on_end() -> None:
            env.robot.reset_targets()
            episode_home = home_for_deployment(
                env, args.home_config, args.home_config_noise
            )
            env.robot.home(blocking=False, home_config=episode_home)
            env.gripper.open()

            logger.info(
                "Waiting for user to decide on success/failure if evaluating."
            )
            if recording_manager.state != "exit":
                evaluator.evaluate(episode=recording_manager.episode_count)

        with evaluator.start_eval(overwrite=True, activate=args.evaluate):
            with recording_manager:
                while not recording_manager.done():
                    logger.info(
                        "→ Episode %s / %s",
                        recording_manager.episode_count + 1,
                        recording_manager.num_episodes,
                    )
                    recording_manager.record_episode(
                        data_fn=policy.make_data_fn(),
                        task="Pick up the lego block.",
                        on_start=on_start,
                        on_end=on_end,
                    )
                    logger.info("Episode finished.")

        logger.info("Shutting down inference process.")
        policy.shutdown()
        policy = None

        logger.info("Homing robot after deployment.")
        final_home = home_after_deployment(
            env,
            args.home_config,
            args.after_teleop,
            args.home_config_noise,
        )
        env.home(home_config=final_home)

        logger.info("Closing the environment.")
        env.close()
        env = None
        logger.info("Finished recording.")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception(exc)
        if policy is not None:
            policy.shutdown()
        if env is not None:
            env.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
