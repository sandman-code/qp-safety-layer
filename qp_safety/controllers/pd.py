"""Two-phase PD pushing controller."""

from __future__ import annotations

import numpy as np

from .base import BaseController


class PDPushController(BaseController):
    """
    Two-phase proportional controller for planar pushing.

    Phase 1 — Approach: moves the EE to a standoff position directly behind
    the object along the push direction (object → goal vector).

    Phase 2 — Push: drives the EE at a constant speed along the push direction
    once it is close enough to the object to make contact.

    Both phases produce velocity commands; the QP filter handles safety.
    """

    def __init__(
        self,
        kp: float = 1,
        approach_dist: float = 0.03,
        push_speed: float = 0.15,
        v_max: float = 0.3,
    ):
        self.kp = kp
        self.approach_dist = approach_dist
        self.push_speed = push_speed
        self.v_max = v_max

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        ee_pos = obs[0:2]
        obj_pos = obs[2:4]
        goal_pos = obs[6:8]
        print(goal_pos)

        to_goal = goal_pos - obj_pos
        dist_to_goal = np.linalg.norm(to_goal)
        if dist_to_goal < 1e-6:
            return np.zeros(2)

        push_dir = to_goal / dist_to_goal
        approach_target = obj_pos - push_dir * self.approach_dist

        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)

        if ee_to_obj > self.approach_dist * 1.05:
            v = self.kp * (approach_target - ee_pos)
        else:
            v = self.push_speed * push_dir

        return np.clip(v, -self.v_max, self.v_max)

    def reset(self) -> None:
        pass
