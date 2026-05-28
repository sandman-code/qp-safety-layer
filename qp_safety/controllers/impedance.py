"""Cartesian impedance controller for planar pushing."""

from __future__ import annotations

import numpy as np

from .base import BaseController


class ImpedanceController(BaseController):
    """
    Cartesian impedance controller for planar pushing.

    Models the EE as a virtual mass-spring-damper system:
        F = K * (x_d - x) - D * x_dot
        v_cmd = F / D                  [inertia-free, velocity-level output]

    The desired position x_d uses the same two-phase approach logic as
    PDPushController so the two controllers are directly comparable.

    The key difference from a pure P controller is the explicit velocity
    feedback term: D * x_dot damps oscillations during approach.
    """

    def __init__(
        self,
        stiffness: float = 5.0,
        damping: float = 2.0,
        approach_dist: float = 0.07,
        contact_overshoot: float = 0.01,
        v_max: float = 0.3,
        ctrl_dt: float = 0.05,
    ):
        self.K = stiffness
        self.D = damping
        self.approach_dist = approach_dist
        self.contact_overshoot = contact_overshoot
        self.v_max = v_max
        self.ctrl_dt = ctrl_dt
        self._prev_ee_pos: np.ndarray | None = None

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        ee_pos = obs[0:2]
        obj_pos = obs[2:4]
        goal_pos = obs[6:8]

        # EE velocity via finite differences
        if self._prev_ee_pos is None:
            ee_vel = np.zeros(2)
        else:
            ee_vel = (ee_pos - self._prev_ee_pos) / self.ctrl_dt
        self._prev_ee_pos = ee_pos.copy()

        to_goal = goal_pos - obj_pos
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return np.zeros(2)

        push_dir = to_goal / dist
        ee_to_obj = np.linalg.norm(ee_pos - obj_pos)

        if ee_to_obj > self.approach_dist * 1.05:
            x_d = obj_pos - push_dir * self.approach_dist
        else:
            # Slight overshoot to maintain contact pressure
            x_d = obj_pos + push_dir * self.contact_overshoot

        pos_err = x_d - ee_pos
        v_cmd = (self.K * pos_err - self.D * ee_vel) / self.D

        return np.clip(v_cmd, -self.v_max, self.v_max)

    def reset(self) -> None:
        self._prev_ee_pos = None
