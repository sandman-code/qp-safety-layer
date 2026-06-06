"""Velocity-level QP safety filter using OSQP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class QPFilterConfig:
    n_dof: int = 2
    v_max: float = 0.3
    alpha: float = (
        5.0  # velocity constraint decay rate [1/s]: v_toward_wall <= alpha * gap
    )
    d_margin: float = 0.03  # safety margin inset from workspace boundary [m]
    osqp_verbose: bool = False
    osqp_eps_abs: float = 1e-6
    osqp_eps_rel: float = 1e-6
    osqp_max_iter: int = 4000


class QPVelocityFilter:
    """
    Velocity-level safety filter formulated as a convex QP.

    Finds the minimum-perturbation safe velocity given a nominal command:

        min_{v ∈ R^n}  0.5 * ||v - v_nom||^2
        s.t.           A_ws(x) v <= b_ws(x)      workspace boundary constraints
                       -v_max <= v_i <= v_max     per-axis velocity limits

    Workspace constraints use a linear distance-based barrier:
        For each table wall with toward-wall normal n_i:
            n_i · v <= alpha * max(d_i(x) - d_margin, 0)
        where d_i(x) is the current gap between the EE and that wall.

    When the EE is within d_margin of a wall, velocity toward that wall
    is forced to zero.  alpha scales how quickly the limit relaxes as the
    EE moves away from the boundary.

    The OSQP problem is set up once and warm-started on subsequent calls
    by updating only q (linear cost) and the constraint bounds.
    """

    def __init__(self, config: QPFilterConfig, workspace: dict[str, float]):
        """
        Args:
            config:    QP filter hyperparameters
            workspace: table bounds {'x_min', 'x_max', 'y_min', 'y_max'} [m]
        """
        self.cfg = config
        self.workspace = workspace
        self._solver = osqp.OSQP()
        self._setup_done = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, v_nom: np.ndarray, ee_pos: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Project v_nom to the nearest feasible safe velocity.

        Args:
            v_nom:  nominal velocity, shape (n_dof,)
            ee_pos: current EE position, shape (>=2,)

        Returns:
            v_safe:   filtered velocity, shape (n_dof,)
            modified: True when the filter changed the nominal command
        """
        n = self.cfg.n_dof
        v_nom_clipped = v_nom[:n].astype(np.float64)

        P = sp.eye(n, format="csc")
        q = -v_nom_clipped

        A_dense, l_vec, u_vec = self._build_constraints(ee_pos)
        A = sp.csc_matrix(A_dense)

        if not self._setup_done:
            self._solver.setup(
                P,
                q,
                A,
                l_vec,
                u_vec,
                warm_starting=True,
                verbose=self.cfg.osqp_verbose,
                eps_abs=self.cfg.osqp_eps_abs,
                eps_rel=self.cfg.osqp_eps_rel,
                max_iter=self.cfg.osqp_max_iter,
                polishing=False,
            )
            self._setup_done = True
        else:
            self._solver.update(q=q, Ax=A.data, l=l_vec, u=u_vec)

        result = self._solver.solve()

        # status_val 1 = solved, 2 = solved (inaccurate) — both acceptable
        if result.info.status_val not in (1, 2):
            v_safe = np.clip(v_nom_clipped, -self.cfg.v_max, self.cfg.v_max)
        else:
            v_safe = result.x[:n]

        modified = bool(np.linalg.norm(v_safe - v_nom_clipped) > 1e-4)
        return v_safe, modified

    def is_safe(self, v: np.ndarray, ee_pos: np.ndarray) -> bool:
        """Return True if velocity v satisfies all constraints at ee_pos."""
        A, _, u = self._build_constraints(ee_pos)
        return bool(np.all(A @ v[: self.cfg.n_dof] <= u + 1e-6))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_constraints(
        self, ee_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build OSQP constraint matrices l <= A v <= u.

        Rows 0-3: workspace boundary constraints (one per wall)
        Rows 4-5: velocity box constraints
        """
        ws = self.workspace
        alpha = self.cfg.alpha
        dm = self.cfg.d_margin
        v_max = self.cfg.v_max
        n = self.cfg.n_dof
        px, py = float(ee_pos[0]), float(ee_pos[1])

        # Each row is a unit vector pointing TOWARD a wall.
        # Constraint n_i · v <= alpha * gap_i prevents motion into the wall.
        # When gap < d_margin → rhs = 0 → no velocity toward that wall allowed.
        A_ws = np.array(
            [
                [1.0, 0.0],  # right wall  (+x): vx  <= alpha*(x_max - px - dm)
                [-1.0, 0.0],  # left wall   (-x): -vx <= alpha*(px - x_min - dm)
                [0.0, 1.0],  # top wall    (+y): vy  <= alpha*(y_max - py - dm)
                [0.0, -1.0],  # bottom wall (-y): -vy <= alpha*(py - y_min - dm)
            ],
            dtype=np.float64,
        )

        b_ws = np.array(
            [
                alpha * max(ws["x_max"] - px - dm, 0.0),
                alpha * max(px - ws["x_min"] - dm, 0.0),
                alpha * max(ws["y_max"] - py - dm, 0.0),
                alpha * max(py - ws["y_min"] - dm, 0.0),
            ],
            dtype=np.float64,
        )

        A_box = np.eye(n, dtype=np.float64)

        A = np.vstack([A_ws, A_box])
        l = np.hstack([np.full(4, -1e9), np.full(n, -v_max)])
        u = np.hstack([b_ws, np.full(n, v_max)])

        return A, l, u


if __name__ == "__main__":
    import sys
    import numpy as np
    from pathlib import Path

    conf = QPFilterConfig()
    workspace = {"x_min": -0.4, "x_max": 0.4, "y_min": -0.4, "y_max": 0.4}
    qp = QPVelocityFilter(config=conf, workspace=workspace)

    v_nom = np.array([0, 0.25, 0.6])
    ee_pos = np.array([0, 0, 0])
    safe = qp.filter(v_nom=v_nom, ee_pos=ee_pos)
    print(safe)
