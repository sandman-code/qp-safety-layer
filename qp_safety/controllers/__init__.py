from .base import BaseController
from .pd import PDPushController
from .impedance import ImpedanceController

REGISTRY: dict[str, type[BaseController]] = {
    "pd": PDPushController,
    "impedance": ImpedanceController,
}

__all__ = ["BaseController", "PDPushController", "ImpedanceController", "REGISTRY"]
