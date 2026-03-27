import pytest

from nerif.observability.budget import (
    BudgetCallbackHandler,  # noqa: F401
    BudgetConfig,
    BudgetExceededError,
    BudgetManager,
)


def test_budget_no_limits():
    bm = BudgetManager(BudgetConfig())
    bm.record(cost_usd=100.0, tokens=1_000_000)
    assert bm.total_cost_usd == 100.0
    assert bm.total_tokens == 1_000_000


def test_budget_hard_limit_usd():
    bm = BudgetManager(BudgetConfig(hard_limit_usd=1.0))
    bm.record(cost_usd=0.5, tokens=100)
    with pytest.raises(BudgetExceededError) as exc_info:
        bm.record(cost_usd=0.6, tokens=100)
    assert exc_info.value.budget_type == "usd"
    assert exc_info.value.limit == 1.0


def test_budget_hard_limit_tokens():
    bm = BudgetManager(BudgetConfig(hard_limit_tokens=1000))
    bm.record(cost_usd=0.0, tokens=500)
    with pytest.raises(BudgetExceededError):
        bm.record(cost_usd=0.0, tokens=600)


def test_budget_soft_limit_fires_callback():
    warnings = []
    bm = BudgetManager(
        BudgetConfig(soft_limit_usd=1.0),
        on_soft_limit=lambda t, lim, a: warnings.append((t, lim, a)),
    )
    bm.record(cost_usd=0.5, tokens=0)
    assert len(warnings) == 0
    bm.record(cost_usd=0.6, tokens=0)
    assert len(warnings) == 1
    assert warnings[0][0] == "usd"
    bm.record(cost_usd=0.1, tokens=0)
    assert len(warnings) == 1  # no re-fire


def test_budget_remaining():
    bm = BudgetManager(BudgetConfig(hard_limit_usd=5.0, hard_limit_tokens=10000))
    bm.record(cost_usd=1.0, tokens=3000)
    rem = bm.remaining()
    assert rem["usd"] == pytest.approx(4.0)
    assert rem["tokens"] == 7000


def test_budget_reset():
    bm = BudgetManager(BudgetConfig(soft_limit_usd=1.0))
    bm.record(cost_usd=2.0, tokens=500)
    bm.reset()
    assert bm.total_cost_usd == 0.0
    assert bm.total_tokens == 0
