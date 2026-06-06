"""
Evaluate all controllers with and without the QP safety filter.

Produces a table of success rate, mean distance-to-goal, and mean filter
activation rate, suitable for inclusion in a paper results section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qp_safety.controllers import REGISTRY, BaseController
from qp_safety.envs.planar_push import PlanarPushEnv
from qp_safety.safety.qp_filter import QPFilterConfig, QPVelocityFilter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluation of controllers ± QP filter."
    )
    p.add_argument(
        "--controllers", nargs="+", choices=list(REGISTRY), default=list(REGISTRY)
    )
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--v-max", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None, help="Path to write JSON results")
    return p.parse_args()


def evaluate_one(
    controller: BaseController,
    use_qp: bool,
    n_episodes: int,
    max_steps: int,
    v_max: float,
    base_seed: int,
) -> dict[str, Any]:
    env = PlanarPushEnv(max_episode_steps=max_steps, v_max=v_max)

    qp_filter: QPVelocityFilter | None = None
    if use_qp:
        qp_filter = QPVelocityFilter(QPFilterConfig(v_max=v_max), env.workspace)

    episodes = []
    for ep in range(n_episodes):
        controller.reset()
        obs, _ = env.reset(seed=base_seed + ep)
        ep_reward = 0.0
        n_filtered = 0

        for step in range(max_steps):
            v_nom = controller.compute_action(obs)

            if qp_filter is not None:
                v_cmd, modified = qp_filter.filter(v_nom, obs[:2])
                n_filtered += int(modified)
            else:
                v_cmd = v_nom

            obs, reward, terminated, truncated, info = env.step(v_cmd)
            ep_reward += reward

            if terminated or truncated:
                break

        episodes.append(
            {
                "success": bool(info["success"]),
                "dist_to_goal": float(info["dist_to_goal"]),
                "total_reward": float(ep_reward),
                "steps": int(step + 1),
                "filter_rate": float(n_filtered / (step + 1)) if use_qp else 0.0,
            }
        )

    env.close()
    return {
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_dist": float(np.mean([e["dist_to_goal"] for e in episodes])),
        "std_dist": float(np.std([e["dist_to_goal"] for e in episodes])),
        "mean_reward": float(np.mean([e["total_reward"] for e in episodes])),
        "mean_filter_rate": float(np.mean([e["filter_rate"] for e in episodes])),
        "episodes": episodes,
    }


def print_table(all_results: dict) -> None:
    header = f"{'Controller':<16} {'QP':>4}  {'Success':>9}  {'Dist (m)':>10}  {'Filter%':>8}"
    print("\n" + header)
    print("-" * len(header))
    for ctrl_name, variants in all_results.items():
        for use_qp, res in variants.items():
            qp_label = "yes" if use_qp == "with_qp" else "no"
            print(
                f"{ctrl_name:<16} {qp_label:>4}  "
                f"{res['success_rate']:>8.1%}  "
                f"{res['mean_dist']:>8.4f} m  "
                f"{res['mean_filter_rate']:>7.1%}"
            )
    print()


if __name__ == "__main__":
    args = parse_args()
    all_results: dict[str, Any] = {}

    for ctrl_name in args.controllers:
        all_results[ctrl_name] = {}
        for label, use_qp in [("without_qp", False), ("with_qp", True)]:
            print(
                f"  evaluating {ctrl_name:12s} {'+ QP' if use_qp else '    '} ...",
                end=" ",
                flush=True,
            )
            controller = REGISTRY[ctrl_name](v_max=args.v_max)
            res = evaluate_one(
                controller=controller,
                use_qp=use_qp,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                v_max=args.v_max,
                base_seed=args.seed,
            )
            all_results[ctrl_name][label] = res
            print(f"success={res['success_rate']:.1%}  dist={res['mean_dist']:.4f} m")

    print_table(all_results)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {out}")
