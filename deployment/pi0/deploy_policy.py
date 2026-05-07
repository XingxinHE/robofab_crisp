"""Deploy a Pi0 policy with robofab-specific homing controls."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

import crisp_gym  # noqa: F401
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging

from deployment.pi0.pi0_policy import Pi0LerobotPolicy
from teleoperations.gamepad.home_config import get_gamepad_home_config


DEFAULT_TASK = "pick and throw the paper with the tray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a Pi0 policy and record deployment data in LeRobot format"
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
        help="Path to a Pi0 pretrained_model directory.",
    )
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--env-namespace", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Language task prompt passed to Pi0. Keep this aligned with training.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--startup-timeout-sec", type=float, default=900.0)
    parser.add_argument("--action-timeout-sec", type=float, default=120.0)
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


def normalize_model_path(path: str) -> Path:
    model_path = Path(path).expanduser()
    nested = model_path / "pretrained_model"
    if nested.is_dir():
        model_path = nested
    return model_path


def select_model_path(path: str | None, logger: logging.Logger) -> Path:
    if path is not None:
        return normalize_model_path(path)

    logger.info("No path provided. Searching for Pi0 models in 'outputs/train'.")
    models_path = Path("outputs/train")
    if not models_path.exists() or not models_path.is_dir():
        raise FileNotFoundError(
            "'outputs/train' does not exist. Provide a model path with --path."
        )

    models = sorted(
        (
            model
            for model in models_path.glob("**/pretrained_model")
            if is_pi0_model(model)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not models:
        raise FileNotFoundError(
            "No Pi0 pretrained_model directories found under outputs/train. "
            "Provide a model path with --path."
        )

    selected = prompt.prompt(
        message="Please select a Pi0 model to use for deployment:",
        options=[str(model) for model in models],
        default=str(models[0]),
    )
    return normalize_model_path(selected)


def is_pi0_model(model_path: Path) -> bool:
    try:
        config = json.loads((model_path / "config.json").read_text())
    except Exception:  # noqa: BLE001
        return False
    return config.get("type") == "pi0"


def validate_pi0_model(model_path: Path) -> None:
    config_path = model_path / "config.json"
    train_config_path = model_path / "train_config.json"
    if not config_path.is_file() or not train_config_path.is_file():
        raise FileNotFoundError(
            f"Invalid model path: {model_path}. "
            "Expected config.json and train_config.json."
        )

    config = json.loads(config_path.read_text())
    train_config = json.loads(train_config_path.read_text())
    policy_type = config.get("type") or (
        train_config.get("policy") or {}
    ).get("type")
    if policy_type != "pi0":
        raise ValueError(
            f"Expected a Pi0 checkpoint, got policy type {policy_type!r}."
        )


def resolve_prompted_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset:"
        )
        logger.info("Using repository ID: %s", args.repo_id)

    model_path = select_model_path(args.path, logger)
    validate_pi0_model(model_path)
    args.path = str(model_path)
    logger.info("Using Pi0 model path: %s", args.path)

    if args.env_namespace is None:
        args.env_namespace = prompt.prompt(
            "Please enter the follower robot namespace:",
            default="",
        )
        logger.info("Using follower namespace: %s", args.env_namespace or "<root>")

    if args.env_config is None:
        follower_configs = list_env_configs()
        args.env_config = prompt.prompt(
            "Please enter the follower robot configuration name.",
            options=follower_configs,
            default=follower_configs[0],
        )
        logger.info("Using follower configuration: %s", args.env_config)


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
            args.env_config,
            control_type=ctrl_type,
            namespace=args.env_namespace,
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

        logger.info("Setting up the Pi0 policy.")
        policy = Pi0LerobotPolicy(
            pretrained_path=args.path,
            env=env,
            task=args.task,
            warmup_steps=args.warmup_steps,
            startup_timeout_sec=args.startup_timeout_sec,
            action_timeout_sec=args.action_timeout_sec,
        )

        logger.info("Homing robot before starting Pi0 deployment recording.")
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

            logger.info("Waiting for user to decide on success/failure if evaluating.")
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
                    logger.info("▷ Task: %s", args.task)
                    recording_manager.record_episode(
                        data_fn=policy.make_data_fn(),
                        task=args.task,
                        on_start=on_start,
                        on_end=on_end,
                    )
                    logger.info("Episode finished.")

        logger.info("Shutting down Pi0 inference process.")
        policy.shutdown()
        policy = None

        logger.info("Homing robot after Pi0 deployment.")
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
        logger.info("Finished Pi0 deployment recording.")
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
