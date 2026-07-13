"""Token-accounting helpers for the released fixed-budget RCWT runs."""

from __future__ import annotations

from dataclasses import dataclass

MAIN_TASK_INSTRUCTION_TOKENS = 337


@dataclass(frozen=True)
class FixedBudgetAllocation:
    """Realized token allocation for one historical main-task target ratio."""

    target_ratio: float
    total_budget: int
    fixed_task_tokens: int
    coordination_tokens: int
    reference_tokens: int

    @property
    def coordination_share(self) -> float:
        """Return the realized full-budget share ``c/W``."""
        return self.coordination_tokens / self.total_budget

    @property
    def residual_task_tokens(self) -> int:
        """Return fixed task instructions plus residual reference tokens."""
        return self.total_budget - self.coordination_tokens


def fixed_budget_allocation(
    total_budget: int,
    target_ratio: float,
    fixed_task_tokens: int = MAIN_TASK_INSTRUCTION_TOKENS,
) -> FixedBudgetAllocation:
    """Reproduce the historical runner allocation for ``q=c/(W-u)``."""
    if total_budget <= fixed_task_tokens:
        raise ValueError("total_budget must exceed fixed_task_tokens")
    if not 0.0 <= target_ratio < 1.0:
        raise ValueError("target_ratio must be in [0, 1)")

    variable_budget = total_budget - fixed_task_tokens
    coordination_tokens = int(variable_budget * target_ratio)
    reference_tokens = variable_budget - coordination_tokens
    return FixedBudgetAllocation(
        target_ratio=target_ratio,
        total_budget=total_budget,
        fixed_task_tokens=fixed_task_tokens,
        coordination_tokens=coordination_tokens,
        reference_tokens=reference_tokens,
    )
