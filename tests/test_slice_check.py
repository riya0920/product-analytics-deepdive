"""The slice check has to be able to fail, or "it holds everywhere" means nothing.

Two of these tests are the ones that matter: one pins that a genuine effect is
reported as holding, and one pins that a pooled gap known to be pure composition
is reported as collapsing. A detector that only ever prints HOLDS would pass the
first and fail the second, which is exactly the bug the first draft had.
"""
import duckdb
import pytest

from analytics.segments import DB
from analytics.slice_check import slice_check


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(DB, read_only=True)
    yield c
    c.close()


def test_real_channel_effect_holds_in_every_slice(con):
    rep = slice_check(con, focus="organic", baseline="paid_search",
                      metric="conversion", dims=("platform", "region"))
    assert rep["real_reversals"] == 0
    assert rep["slices_vanished"] == 0
    assert rep["slices_holding"] == rep["slices_evaluated"]
    assert rep["summary"].startswith("HOLDS")


def test_pure_composition_gap_is_reported_as_collapsing(con):
    """apac beats emea on conversion, and the generator gives region NO direct
    effect. The entire pooled gap is channel mix, so conditioning on channel must
    make it disappear rather than merely fail to reverse."""
    rep = slice_check(con, by="region", focus="apac", baseline="emea",
                      metric="conversion", dims=("platform", "channel"))
    assert rep["headline"]["significant"], "the pooled gap should look real before slicing"
    assert rep["slices_vanished"] > rep["slices_holding"]
    assert rep["summary"].startswith("COLLAPSES")


def test_underpowered_slices_are_not_counted_as_agreement(con):
    rep = slice_check(con, focus="organic", baseline="paid_search",
                      metric="conversion", dims=("platform", "region"))
    for group in rep["groups"]:
        for row in group["rows"]:
            if row["verdict"] == "underpowered":
                assert not row["powered"]
    assert rep["slices_powered"] <= rep["slices_evaluated"]


def test_thin_cells_are_suppressed_not_reported(con):
    rep = slice_check(con, focus="organic", baseline="paid_search",
                      metric="conversion", dims=("platform", "region"))
    for group in rep["groups"]:
        for row in group["rows"]:
            if row["verdict"] == "suppressed":
                assert row["min_n"] < rep["min_cell"]


def test_cannot_slice_by_the_column_being_compared(con):
    with pytest.raises(ValueError):
        slice_check(con, by="channel", focus="organic", baseline="paid_search",
                    dims=("channel", "region"))


def test_unknown_metric_is_rejected(con):
    with pytest.raises(ValueError):
        slice_check(con, metric="not_a_metric")
