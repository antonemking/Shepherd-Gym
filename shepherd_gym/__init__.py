"""shepherd-gym — a low-stress herding RL environment (PufferLib-ready)."""
from .env import ShepherdEnv, ShepherdConfig
from . import baselines, render

__all__ = ["ShepherdEnv", "ShepherdConfig", "baselines", "render"]
__version__ = "0.1.0"
