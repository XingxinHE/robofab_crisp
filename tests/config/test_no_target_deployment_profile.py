from __future__ import annotations

from pathlib import Path
import sys
import tomllib

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_env_config(name: str) -> dict:
    with (ROOT / "config" / "envs" / f"{name}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_no_target_gamepad_deploy_configs_exclude_controller_target_state() -> None:
    for name in [
        "fr3_3cams_gamepad_deploy_streamed_no_target",
        "fr3_3cams_gamepad_deploy_streamed_joint_states_fallback_no_target",
    ]:
        config = _load_env_config(name)

        assert config["observations_to_include_to_state"] == [
            "observation.state.cartesian",
            "observation.state.gripper",
            "observation.state.joints",
        ]


def test_no_target_preflight_selects_no_target_env_config_names() -> None:
    from deployment.act import preflight_fr3_3cams_gamepad_no_target as preflight

    assert preflight.PRIMARY_ENV_CONFIG == "fr3_3cams_gamepad_deploy_streamed_no_target"
    assert (
        preflight.FALLBACK_ENV_CONFIG
        == "fr3_3cams_gamepad_deploy_streamed_joint_states_fallback_no_target"
    )


def test_pixi_exposes_no_target_gamepad_act_deployment_task() -> None:
    with (ROOT / "pixi.toml").open("rb") as f:
        pixi = tomllib.load(f)

    preflight_task = pixi["tasks"]["preflight-deploy-fr3-3cams-gamepad-no-target"]
    assert preflight_task == (
        "python -m deployment.act.preflight_fr3_3cams_gamepad_no_target"
    )

    task = pixi["tasks"]["deploy-act-fr3-3cams-gamepad-no-target"]
    assert "deployment.act.preflight_fr3_3cams_gamepad_no_target" in task
    assert "local/fr3_gamepad_3cams_deploy_no_target" in task
