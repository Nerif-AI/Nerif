"""Budget management with soft and hard limits."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Set

from ..exceptions import NerifError

LOGGER_NAME = "nerif"
_LOGGER = logging.getLogger(f"{LOGGER_NAME}.budget")


class BudgetExceededError(NerifError):
    def __init__(self, budget_type: str, limit: float, actual: float):
        self.budget_type = budget_type
        self.limit = limit
        self.actual = actual
        super().__init__(f"Budget exceeded: {budget_type} limit {limit}, actual {actual}")


@dataclass
class BudgetConfig:
    soft_limit_usd: Optional[float] = None
    hard_limit_usd: Optional[float] = None
    soft_limit_tokens: Optional[int] = None
    hard_limit_tokens: Optional[int] = None


class BudgetManager:
    def __init__(
        self,
        config: BudgetConfig,
        on_soft_limit: Optional[Callable[[str, float, float], None]] = None,
    ):
        self.config = config
        self.on_soft_limit = on_soft_limit
        self._total_cost_usd: float = 0.0
        self._total_tokens: int = 0
        self._soft_triggered: Set[str] = set()

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def record(self, cost_usd: float, tokens: int) -> None:
        self._total_cost_usd += cost_usd
        self._total_tokens += tokens
        self._check_limits()

    def _check_limits(self) -> None:
        # Hard limits (raise)
        if self.config.hard_limit_usd is not None and self._total_cost_usd > self.config.hard_limit_usd:
            raise BudgetExceededError("usd", self.config.hard_limit_usd, self._total_cost_usd)
        if self.config.hard_limit_tokens is not None and self._total_tokens > self.config.hard_limit_tokens:
            raise BudgetExceededError("tokens", self.config.hard_limit_tokens, self._total_tokens)
        # Soft limits (callback + warning, once per type)
        if (
            self.config.soft_limit_usd is not None
            and self._total_cost_usd > self.config.soft_limit_usd
            and "usd" not in self._soft_triggered
        ):
            self._soft_triggered.add("usd")
            _LOGGER.warning(
                "Soft budget limit reached: usd limit %.4f, actual %.4f",
                self.config.soft_limit_usd,
                self._total_cost_usd,
            )
            if self.on_soft_limit:
                self.on_soft_limit("usd", self.config.soft_limit_usd, self._total_cost_usd)
        if (
            self.config.soft_limit_tokens is not None
            and self._total_tokens > self.config.soft_limit_tokens
            and "tokens" not in self._soft_triggered
        ):
            self._soft_triggered.add("tokens")
            _LOGGER.warning(
                "Soft budget limit reached: tokens limit %d, actual %d",
                self.config.soft_limit_tokens,
                self._total_tokens,
            )
            if self.on_soft_limit:
                self.on_soft_limit("tokens", self.config.soft_limit_tokens, self._total_tokens)

    def remaining(self) -> dict:
        result = {}
        if self.config.hard_limit_usd is not None:
            result["usd"] = self.config.hard_limit_usd - self._total_cost_usd
        elif self.config.soft_limit_usd is not None:
            result["usd"] = self.config.soft_limit_usd - self._total_cost_usd
        if self.config.hard_limit_tokens is not None:
            result["tokens"] = self.config.hard_limit_tokens - self._total_tokens
        elif self.config.soft_limit_tokens is not None:
            result["tokens"] = self.config.soft_limit_tokens - self._total_tokens
        return result

    def reset(self) -> None:
        self._total_cost_usd = 0.0
        self._total_tokens = 0
        self._soft_triggered.clear()


class BudgetCallbackHandler:
    def __init__(self, budget: BudgetManager):
        self.budget = budget

    def on_llm_end(self, event) -> None:
        tokens = event.prompt_tokens + event.completion_tokens
        self.budget.record(cost_usd=event.cost_usd, tokens=tokens)
