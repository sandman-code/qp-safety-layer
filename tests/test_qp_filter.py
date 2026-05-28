"""Unit tests for the QP velocity safety filter."""

import numpy as np
import pytest

from qp_safety.safety.qp_filter import QPFilterConfig, QPVelocityFilter

WORKSPACE = {"x_min": -0.38, "x_max": 0.38, "y_min": -0.38, "y_max": 0.38}


@pytest.fixture
def filt() -> QPVelocityFilter:
    cfg = QPFilterConfig(v_max=0.3, alpha=5.0, d_margin=0.03)
    return QPVelocityFilter(cfg, WORKSPACE)


def test_safe_velocity_passes_through(filt):
    """Small velocity at workspace center should be returned unchanged."""
    v_nom = np.array([0.10, 0.05])
    v_safe, modified = filt.filter(v_nom, np.array([0.0, 0.0]))
    assert not modified
    np.testing.assert_allclose(v_safe, v_nom, atol=1e-3)


def test_velocity_toward_wall_is_reduced(filt):
    """Velocity into a wall at close range must be reduced."""
    v_nom = np.array([0.30, 0.0])   # moving right at v_max
    ee_pos = np.array([0.36, 0.0])  # 0.02 m gap — inside d_margin=0.03
    v_safe, modified = filt.filter(v_nom, ee_pos)
    assert modified
    assert v_safe[0] < v_nom[0] - 1e-3


def test_velocity_away_from_wall_unconstrained(filt):
    """Velocity away from the nearest wall should not be constrained."""
    v_nom = np.array([-0.30, 0.0])  # moving left (away from right wall)
    ee_pos = np.array([0.36, 0.0])  # near right wall
    v_safe, modified = filt.filter(v_nom, ee_pos)
    assert not modified
    np.testing.assert_allclose(v_safe, v_nom, atol=1e-3)


def test_output_respects_v_max(filt):
    """Filtered velocity must satisfy the v_max box constraint."""
    rng = np.random.default_rng(42)
    for _ in range(100):
        v_nom = rng.uniform(-0.6, 0.6, size=2)
        ee_pos = rng.uniform(-0.30, 0.30, size=2)
        v_safe, _ = filt.filter(v_nom, ee_pos)
        assert np.all(np.abs(v_safe) <= 0.3 + 1e-4), f"v_safe={v_safe} exceeds v_max"


def test_is_safe_agrees_with_filter(filt):
    """is_safe predicate should be consistent with the filter output."""
    v_interior = np.array([0.1, 0.0])
    v_over_limit = np.array([0.5, 0.0])  # exceeds v_max
    ee = np.array([0.0, 0.0])
    assert filt.is_safe(v_interior, ee)
    assert not filt.is_safe(v_over_limit, ee)


def test_warm_start_consistency(filt):
    """Results should be identical with and without warm-starting."""
    v_nom = np.array([0.20, 0.10])
    ee_pos = np.array([0.0, 0.0])
    v1, _ = filt.filter(v_nom, ee_pos)
    v2, _ = filt.filter(v_nom, ee_pos)  # second call uses warm-start
    np.testing.assert_allclose(v1, v2, atol=1e-5)


def test_corner_near_two_walls(filt):
    """EE near a corner should be constrained on both axes."""
    v_nom = np.array([0.30, 0.30])
    ee_pos = np.array([0.37, 0.37])  # near (+x, +y) corner
    v_safe, modified = filt.filter(v_nom, ee_pos)
    assert modified
    assert v_safe[0] < v_nom[0]
    assert v_safe[1] < v_nom[1]
