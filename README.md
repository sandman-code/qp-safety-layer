# QP Safety Layer for Planar Pushing

A velocity-level quadratic programming (QP) safety filter applied to planar pushing
with a UR5e end-effector in MuJoCo simulation.  The filter projects any nominal
controller's velocity command onto the nearest feasible safe velocity, enforcing
workspace boundary constraints with zero reliance on the nominal controller's design.

---

## Method Overview

Given a nominal EE velocity `v_nom` from any controller, the safety filter solves:

```
min_{v ∈ R²}   0.5 ‖v − v_nom‖²
s.t.           nᵢ · v ≤ α · max(dᵢ(x) − δ, 0)    ∀ workspace wall i
               ‖v‖_∞ ≤ v_max
```

where `dᵢ(x)` is the gap between the EE and wall `i`, `δ` is a safety margin,
and `α` is a decay-rate gain.  This is a strictly convex QP solved in real time
with [OSQP](https://osqp.org/).

The filter is **controller-agnostic**: any nominal policy can be plugged in and
evaluated behind the same safety layer.

---

## Project Structure

```
qp-safety-layer/
├── assets/
│   └── scene/
│       └── planar_push.xml      MuJoCo scene (table, EE puck, box, goal site)
├── configs/
│   ├── default.yaml             env + QP filter + eval hyperparameters
│   └── controller/
│       ├── pd.yaml
│       └── impedance.yaml
├── qp_safety/
│   ├── envs/
│   │   └── planar_push.py       Gymnasium env — mocap EE, free-body box
│   ├── safety/
│   │   └── qp_filter.py         QPVelocityFilter (OSQP, warm-started)
│   ├── controllers/
│   │   ├── base.py              BaseController interface
│   │   ├── pd.py                Two-phase PD approach-and-push
│   │   └── impedance.py         Cartesian impedance (spring-damper)
│   └── utils/
│       └── math_utils.py
├── scripts/
│   ├── run_episode.py           Single episode with optional QP filter
│   └── evaluate.py              Batch evaluation → results table + JSON
├── tests/
│   ├── test_qp_filter.py
│   └── test_env.py
└── sandbox/                     Personal exploration (not part of the paper)
    ├── NTK.py
    ├── qp-ex.py
    └── mujoco-tutorial.py
```

---

## Installation

```bash
git clone <repo-url>
cd qp-safety-layer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and MuJoCo ≥ 3.0.

---

## Quick Start

**Run a single episode:**
```bash
# PD controller, no filter
python scripts/run_episode.py --controller pd

# Impedance controller with QP safety filter
python scripts/run_episode.py --controller impedance --use-qp
```

**Batch evaluation (all controllers ± QP):**
```bash
python scripts/evaluate.py --n-episodes 50 --save results/results.json
```

Sample output:
```
Controller       QP    Success    Dist (m)   Filter%
-------------------------------------------------------
pd               no    xx.x%    0.xxxx m      0.0%
pd              yes    xx.x%    0.xxxx m     xx.x%
impedance        no    xx.x%    0.xxxx m      0.0%
impedance       yes    xx.x%    0.xxxx m     xx.x%
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## Adding a New Controller

1. Create `qp_safety/controllers/my_controller.py` with a class that extends `BaseController`.
2. Implement `compute_action(obs) → np.ndarray` and optionally `reset()`.
3. Register it in `qp_safety/controllers/__init__.py`:
   ```python
   from .my_controller import MyController
   REGISTRY["my_controller"] = MyController
   ```
4. Add a config file `configs/controller/my_controller.yaml`.

The controller will automatically appear in `--controller` choices for both scripts.

---

## Environment Details

| Property | Value |
|---|---|
| Observation dim | 9 (ee_xy, obj_xy, obj_yaw, goal_xy, rel_xy) |
| Action dim | 2 (EE velocity xy [m/s]) |
| Control frequency | 20 Hz (configurable) |
| Physics timestep | 2 ms (500 Hz) |
| Table workspace | ±0.38 m (x and y) |
| Success threshold | 0.02 m object-to-goal distance |

---

## Citation

```bibtex
@article{author2026qpsafety,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {2026},
}
```
