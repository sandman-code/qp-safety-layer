"""Run a single episode and print a result summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import mujoco
import mujoco.viewer

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qp_safety.controllers import REGISTRY
from qp_safety.envs.planar_push import PlanarPushEnv
from qp_safety.safety.qp_filter import QPFilterConfig, QPVelocityFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one episode of planar pushing.")
    p.add_argument("--controller", choices=list(REGISTRY), default="pd")
    p.add_argument("--use-qp", action="store_true", help="Enable QP safety filter")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--v-max", type=float, default=0.3)
    p.add_argument(
        "--render", action="store_true", help="Render overhead view to display"
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    env = PlanarPushEnv(
        max_episode_steps=args.max_steps,
        v_max=args.v_max,
        render_mode="human" if args.render else None,
    )
    controller = REGISTRY[args.controller](v_max=args.v_max)

    qp_filter: QPVelocityFilter | None = None
    if args.use_qp:
        qp_filter = QPVelocityFilter(QPFilterConfig(v_max=args.v_max), env.workspace)

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    n_filtered = 0
    step = 0

    for step in range(args.max_steps):
        v_nom = controller.compute_action(obs)

        if qp_filter is not None:
            v_cmd, modified = qp_filter.filter(v_nom, obs[:2])
            n_filtered += int(modified)
        else:
            v_cmd = v_nom

        obs, reward, terminated, truncated, info = env.step(v_cmd)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return {
        "success": info["success"],
        "dist_to_goal": info["dist_to_goal"],
        "total_reward": total_reward,
        "steps": step + 1,
        "n_filtered": n_filtered,
        "filter_rate": n_filtered / (step + 1),
    }


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    label = f"{args.controller}" + (" + QP" if args.use_qp else "")
    print(f"\n=== Episode Result  [{label}]  seed={args.seed} ===")
    for k, v in result.items():
        print(f"  {k:<20s}: {v:.4f}" if isinstance(v, float) else f"  {k:<20s}: {v}")
