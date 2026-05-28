"""Abstract controller interface."""

from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    """
    Common interface for nominal controllers.

    All controllers receive the PlanarPushEnv observation vector and return
    a 2-D EE velocity command [m/s].  The QP safety filter is applied
    downstream — controllers should not clip their own outputs.
    """

    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            obs: (9,) observation from PlanarPushEnv

        Returns:
            v_nom: (2,) nominal EE velocity [m/s]
        """

    def reset(self) -> None:
        """Called at the start of each episode to clear any internal state."""
