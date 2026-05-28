"""Smoke tests for PlanarPushEnv."""

import numpy as np
import pytest

from qp_safety.envs.planar_push import ACT_DIM, OBS_DIM, PlanarPushEnv


@pytest.fixture
def env():
    e = PlanarPushEnv(
        max_episode_steps=10,
        randomize_goal=False,
        randomize_object=False,
    )
    yield e
    e.close()


def test_obs_shape_and_dtype(env):
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_action_space_shape(env):
    assert env.action_space.shape == (ACT_DIM,)


def test_step_returns_finite_reward(env):
    env.reset()
    _, reward, *_ = env.step(np.zeros(ACT_DIM))
    assert np.isfinite(reward)


def test_large_action_is_clipped(env):
    env.reset()
    # action far outside [-v_max, v_max] should not crash
    obs, reward, *_ = env.step(np.array([100.0, 100.0]))
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(reward)


def test_truncation_at_max_steps(env):
    env.reset()
    truncated = False
    for _ in range(15):  # max_episode_steps=10
        _, _, terminated, truncated, _ = env.step(np.zeros(ACT_DIM))
        if terminated or truncated:
            break
    assert truncated


def test_reset_reproducibility(env):
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)


def test_workspace_bounds(env):
    ws = env.workspace
    required_keys = {"x_min", "x_max", "y_min", "y_max"}
    assert required_keys == set(ws.keys())
    assert ws["x_max"] > ws["x_min"]
    assert ws["y_max"] > ws["y_min"]


def test_ee_pos_within_workspace_after_steps(env):
    obs, _ = env.reset()
    ws = env.workspace
    for _ in range(10):
        # Try to drive EE out of bounds
        obs, _, _, _, _ = env.step(np.array([1.0, 1.0]))

    ee = env.ee_pos
    assert ws["x_min"] <= ee[0] <= ws["x_max"]
    assert ws["y_min"] <= ee[1] <= ws["y_max"]
