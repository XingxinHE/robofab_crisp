#!/usr/bin/env python3
"""Train PI0 policy on FR3 leader/follower datasets.

Thin Python wrapper around ``python -m lerobot.scripts.train`` with
PI0 defaults.  Replaces the shell-script pattern used for ACT.

Usage:
    python training/pi0/train.py --repo-id local/fr3_leader_follower_3cams_open_fix_feat
    python training/pi0/train.py --prepare-from local/fr3_leader_follower_3cams_open --smoke
    python training/pi0/train.py --policy-path lerobot/pi0 --steps 100000
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro
import tyro.conf


@dataclass
class TrainArgs:
    # ── Dataset ────────────────────────────────────────────────────────
    repo_id: str = "local/fr3_leader_follower_3cams_open_fix_feat"
    prepare_from: str | None = None

    # ── Training ───────────────────────────────────────────────────────
    steps: int = 50_000
    batch_size: int = 8
    save_freq: int = 10_000
    log_freq: int = 100
    eval_freq: int = 0

    # ── PI0 policy ─────────────────────────────────────────────────────
    policy_path: str | None = None
    lr: float = 2.5e-5
    attention_implementation: str = "eager"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    num_steps: int = 10

    # ── Runtime ────────────────────────────────────────────────────────
    gpu: int = 0
    smoke: bool = False

    # ── Passthrough to lerobot ─────────────────────────────────────────
    extra_args: list[str] = field(
        default_factory=list,
        metadata={tyro.conf.arg: True, "help": "Forwarded to lerobot.scripts.train."},
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATASET_SCRIPT = REPO_ROOT / "dataset" / "00_crisp_to_lerobot_compatible.py"

# lerobot (dacd1d7f) PI0 code accesses GemmaForCausalLM attributes directly
# (embed_tokens, layers, norm, rotary_emb, etc.) that live on the inner
# GemmaModel (self.model) in all recent transformers versions.  We monkey-patch
# GemmaForCausalLM.__getattr__ to delegate to self.model for missing attrs.
_TRANSFORMERS_SHIM = """\
import transformers, torch.nn as nn
_nn_getattr = nn.Module.__getattr__
def _compat(self, name):
    try: return _nn_getattr(self, name)
    except AttributeError:
        m = self._modules.get('model')
        if m is not None and hasattr(m, name):
            return getattr(m, name)
        raise
transformers.GemmaForCausalLM.__getattr__ = _compat
"""


def main(args: TrainArgs) -> None:
    # -- prepare dataset if requested --
    if args.prepare_from is not None:
        dst = args.repo_id
        print(f"[train-pi0] preparing dataset: {args.prepare_from} -> {dst}")
        subprocess.check_call(
            [sys.executable, str(DATASET_SCRIPT),
             "--src-repo-id", args.prepare_from,
             "--dst-repo-id", dst],
        )

    # -- smoke overrides --
    if args.smoke:
        args.steps = 2_000
        args.save_freq = 1_000

    # -- log --
    policy_desc = args.policy_path or "pi0 (from scratch)"
    print(
        f"[train-pi0] policy={policy_desc}  dataset={args.repo_id}  "
        f"steps={args.steps}  batch={args.batch_size}  lr={args.lr}  "
        f"attn={args.attention_implementation}  num_steps={args.num_steps}"
    )

    # -- build lerobot CLI args --
    import os

    lerobot_args: list[str] = [
        f"--dataset.repo_id={args.repo_id}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--dataset.video_backend=pyav",
        "--policy.push_to_hub=false",
        f"--optimizer.lr={args.lr}",
        f"--policy.num_steps={args.num_steps}",
        f"--policy.attention_implementation={args.attention_implementation}",
        f"--policy.freeze_vision_encoder={'true' if args.freeze_vision_encoder else 'false'}",
        f"--policy.train_expert_only={'true' if args.train_expert_only else 'false'}",
    ]

    if args.policy_path is not None:
        lerobot_args.append(f"--policy.path={args.policy_path}")
    else:
        lerobot_args.append("--policy.type=pi0")

    lerobot_args.extend(args.extra_args)

    # -- exec via subprocess with transformers shim --
    wrapper = (
        _TRANSFORMERS_SHIM
        + "import sys, runpy\n"
        + "runpy.run_module('lerobot.scripts.train', run_name='__main__')\n"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    final_cmd = [sys.executable, "-c", wrapper] + lerobot_args
    sys.exit(subprocess.run(final_cmd, env=env).returncode)


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)
