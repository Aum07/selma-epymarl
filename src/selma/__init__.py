"""SELMA: Selective LLM-Guided Multi-Agent Reinforcement Learning.

Stage A exposes structured, immutable LBF load-event records for online
consumption by the episode runner. No detector, provider, potential, or reward
shaping lives here yet.
"""

from .load_events import build_selma_step_info  # noqa: F401
from .types import (  # noqa: F401
    AgentSnapshot,
    FailedLoadEvent,
    FoodSnapshot,
    SelmaStepInfo,
    SuccessfulLoadEvent,
)
