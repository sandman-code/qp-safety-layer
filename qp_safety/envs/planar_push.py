"""Planar pushing environment with a mocap-driven EE puck in MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"

# Observation layout: [ee_x, ee_y, obj_x, obj_y, obj_yaw, goal_x, goal_y, rel_x, rel_y]
OBS_DIM = 9
ACT_DIM = 2


class PlanarPushEnv(gym.Env):
    """
    Planar pushing environment for a velocity-controlled EE puck in MuJoCo.

    The end-effector is a kinematic (mocap) cylindrical puck on a table.
    Velocity commands are integrated to update its XY position each control step,
    then MuJoCo propagates contact dynamics to push the free box object.

    Observation (9-dim):
        ee_pos   (2,)  EE xy position [m]
        obj_pos  (2,)  Object xy position [m]
        obj_yaw  (1,)  Object yaw angle [rad]
        goal_pos (2,)  Goal xy position [m]
        rel_pos  (2,)  goal_pos - obj_pos [m]

    Action (2-dim):
        v_ee  EE velocity command [m/s] in the table XY plane, clipped to [-v_max, v_max]
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        model_path: str | None = None,
        max_episode_steps: int = 500,
        goal_tolerance: float = 0.02,
        v_max: float = 0.3,
        ctrl_dt: float = 0.05,
        render_mode: str | None = None,
        randomize_goal: bool = True,
        randomize_object: bool = True,
    ):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.v_max = v_max
        self.ctrl_dt = ctrl_dt
        self.render_mode = render_mode
        self.randomize_goal = randomize_goal
        self.randomize_object = randomize_object

        xml_path = model_path or str(ASSETS_DIR / "scene" / "planar_push.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._n_substeps = max(1, round(ctrl_dt / self.model.opt.timestep))

        self._ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee")
        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self._goal_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "goal"
        )
        self._ee_mocap_id = int(self.model.body_mocapid[self._ee_body_id])

        obj_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint"
        )
        self._obj_qpos_adr = int(self.model.jnt_qposadr[obj_jnt_id])

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-v_max, high=v_max, shape=(ACT_DIM,), dtype=np.float32
        )

        self._step_count: int = 0
        self._goal_pos = np.array([0.15, 0.10], dtype=np.float64)
        self._renderer: mujoco.Renderer | None = None
        self.viewer: mujoco.viewer.Handle | None = None

        # Table surface z — EE and object centers rest at this height
        self._table_z: float = 0.435

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.randomize_object:
            obj_xy = self.np_random.uniform([-0.08, -0.08], [0.08, 0.08])
            obj_yaw = float(self.np_random.uniform(-np.pi, np.pi))
        else:
            obj_xy = np.array([0.05, 0.05])
            obj_yaw = 0.0

        # Set object freejoint qpos: [x, y, z,  qw, qx, qy, qz]
        adr = self._obj_qpos_adr
        self.data.qpos[adr : adr + 3] = [obj_xy[0], obj_xy[1], self._table_z]
        self.data.qpos[adr + 3] = np.cos(obj_yaw / 2.0)
        self.data.qpos[adr + 4 : adr + 7] = [0.0, 0.0, np.sin(obj_yaw / 2.0)]

        # Place EE behind the object along the -x axis
        ee_init = np.array([obj_xy[0] - 0.12, obj_xy[1], self._table_z])
        self.data.mocap_pos[self._ee_mocap_id] = ee_init

        if self.randomize_goal:
            ws = self.workspace
            for _ in range(100):
                gx = self.np_random.uniform(ws["x_min"] + 0.05, ws["x_max"] - 0.05)
                gy = self.np_random.uniform(ws["y_min"] + 0.05, ws["y_max"] - 0.05)
                g = np.array([gx, gy])
                if np.linalg.norm(g - obj_xy) >= 0.08:
                    self._goal_pos = g
                    break
        else:
            self._goal_pos = np.array([0.15, 0.10])

        self._update_goal_site()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        v_cmd = np.clip(action, -self.v_max, self.v_max).astype(np.float64)
        self._apply_ee_velocity(v_cmd)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        dist = float(np.linalg.norm(self._obj_pos()[:2] - self._goal_pos))
        reward = -dist
        terminated = dist < self.goal_tolerance
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        info: dict[str, Any] = {
            "success": terminated,
            "dist_to_goal": dist,
            "ee_pos": self.ee_pos.tolist(),
            "obj_pos": self._obj_pos()[:2].tolist(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        self._renderer.update_scene(self.data, camera="overhead")
        self.viewer.sync()
        return self._renderer.render()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()

        if self._renderer:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> dict[str, float]:
        """Table workspace bounds in world XY [m], inset from geom edges."""
        return {"x_min": -0.38, "x_max": 0.38, "y_min": -0.38, "y_max": 0.38}

    @property
    def ee_pos(self) -> np.ndarray:
        """Current EE xy position [m]."""
        return self.data.mocap_pos[self._ee_mocap_id, :2].copy()

    @property
    def obj_pos(self) -> np.ndarray:
        """Current object xy position [m]."""
        return self._obj_pos()[:2]

    @property
    def goal_pos(self) -> np.ndarray:
        return self._goal_pos.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.mocap_pos[self._ee_mocap_id, :2].astype(np.float32)
        obj_pos = self._obj_pos()[:2].astype(np.float32)
        obj_yaw = np.float32(self._obj_yaw())
        goal = self._goal_pos.astype(np.float32)
        rel = goal - obj_pos
        return np.concatenate([ee_pos, obj_pos, [obj_yaw], goal, rel])

    def _obj_pos(self) -> np.ndarray:
        return self.data.xpos[self._obj_body_id].copy()

    def _obj_yaw(self) -> float:
        w, x, y, z = self.data.xquat[self._obj_body_id]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _apply_ee_velocity(self, v_xy: np.ndarray) -> None:
        """Integrate velocity command into the mocap EE position."""
        pos = self.data.mocap_pos[self._ee_mocap_id, :2].copy()
        pos += v_xy * self.ctrl_dt
        ws = self.workspace
        pos[0] = np.clip(pos[0], ws["x_min"], ws["x_max"])
        pos[1] = np.clip(pos[1], ws["y_min"], ws["y_max"])
        self.data.mocap_pos[self._ee_mocap_id, :2] = pos

    def _update_goal_site(self) -> None:
        self.model.site_pos[self._goal_site_id, :2] = self._goal_pos
