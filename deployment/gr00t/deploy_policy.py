"""Deploy a remote GR00T policy with robofab-specific homing controls."""

from __future__ import annotations

import argparse
import datetime
import logging

import crisp_gym  # noqa: F401
from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.envs.manipulator_env_config import list_env_configs
from crisp_gym.record.evaluate import Evaluator
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging

from deployment.gr00t.constants import DEFAULT_TASK
from deployment.gr00t.gr00t_remote_policy import Gr00tRemotePolicy
from teleoperations.gamepad.home_config import get_gamepad_home_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a GR00T policy through a remote inference server"
    )
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--robot-type", type=str, default="franka")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--push-to-hub", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--recording-manager-type", type=str, default="keyboard")
    parser.add_argument(
        "--joint-control",
        action="store_true",
        help="Unsupported for GR00T deployment; retained only to fail clearly.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--env-namespace", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument(
        "--groot-transport",
        choices=["http", "zmq"],
        default="http",
        help="Transport used to talk to the GR00T inference server.",
    )
    parser.add_argument(
        "--groot-server",
        type=str,
        default="http://127.0.0.1:8000",
        help="HTTP URL of the GR00T inference server.",
    )
    parser.add_argument(
        "--groot-host",
        type=str,
        default="127.0.0.1",
        help="ZMQ host for the GR00T inference server.",
    )
    parser.add_argument(
        "--groot-port",
        type=int,
        default=5555,
        help="ZMQ port for the GR00T inference server.",
    )
    parser.add_argument(
        "--groot-api-token",
        type=str,
        default=None,
        help="Optional API token for GR00T ZMQ requests.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Language task prompt passed to GR00T. Keep this aligned with training.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=1,
        help="Number of GR00T horizon actions to execute before requesting a new horizon.",
    )
    parser.add_argument("--action-timeout-sec", type=float, default=20.0)
    parser.add_argument("--gripper-max-width-m", type=float, default=0.08)
    parser.add_argument("--max-translation-step-m", type=float, default=0.006)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.06)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--async-inference",
        action="store_true",
        default=False,
        help="Prefetch GR00T horizons in a background thread while queued actions execute.",
    )
    parser.add_argument(
        "--prefetch-threshold",
        type=int,
        default=None,
        help=(
            "Start a background GR00T request when queued actions are at or below this count. "
            "Default: about half of --action-chunk-size."
        ),
    )
    parser.add_argument(
        "--log-timing",
        action="store_true",
        default=False,
        help="Log frame-level timing breakdowns for deployment profiling.",
    )
    parser.add_argument("--timing-log-interval", type=int, default=25)
    parser.add_argument(
        "--validate-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check GR00T /health before deployment starts.",
    )
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


def resolve_prompted_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.repo_id is None:
        args.repo_id = prompt.prompt(
            "Please enter the repository ID for the dataset:",
        )
        logger.info("Using repository ID: %s", args.repo_id)

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
    server = args.groot_server if args.groot_transport == "http" else f"{args.groot_host}_{args.groot_port}"
    safe_server = server.replace("/", "_").replace(":", "_")
    return (
        prompt.prompt(
            "Please enter the output file for evaluation results",
            default=f"evaluation_results_gr00t_{safe_server}_{datetime_now}",
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
        if args.joint_control:
            raise ValueError("GR00T deployment currently supports Cartesian CRISP control only.")

        resolve_prompted_args(args, logger)
        evaluation_file = evaluation_output_file(args)

        env = make_env(
            args.env_config,
            control_type="cartesian",
            namespace=args.env_namespace,
        )

        logger.info("Setting up the GR00T remote policy.")
        policy = Gr00tRemotePolicy(
            env=env,
            transport=args.groot_transport,
            server_url=args.groot_server,
            host=args.groot_host,
            port=args.groot_port,
            api_token=args.groot_api_token,
            task=args.task,
            action_chunk_size=args.action_chunk_size,
            action_timeout_sec=args.action_timeout_sec,
            gripper_max_width_m=args.gripper_max_width_m,
            max_translation_step_m=args.max_translation_step_m,
            max_rotation_step_rad=args.max_rotation_step_rad,
            dry_run=args.dry_run,
            validate_server=args.validate_server,
            async_inference=args.async_inference,
            prefetch_threshold=args.prefetch_threshold,
            log_timing=args.log_timing,
            timing_log_interval=args.timing_log_interval,
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

        logger.info("Homing robot before starting GR00T deployment recording.")
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
            policy.reset()
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

        logger.info("Shutting down GR00T client.")
        policy.shutdown()
        policy = None

        logger.info("Homing robot after GR00T deployment.")
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
        logger.info("Finished GR00T deployment recording.")
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
