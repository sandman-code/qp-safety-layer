"""Shared math utilities."""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """2x2 rotation matrix for angle theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def unit_vec(v: np.ndarray) -> np.ndarray:
    """Normalize a vector; returns zeros if the input norm is near zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)
