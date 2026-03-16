"""Tests for the training-serving consistency checker.

All tests use the in-memory online store fallback and a temporary offline store,
so no Redis or Kafka is needed.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.consistency.checker import (
    INCONSISTENCY_RATE_THRESHOLD,
    RELATIVE_DIFF_THRESHOLD,
    ConsistencyChecker,
)
from src.stores.offline_store import OfflineFeatureStore
from src.stores.online_store import OnlineFeatureStore, _FALLBACK_STORE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE = datetime(2024, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clear_fallback():
    _FALLBACK_STORE.clear()
    yield
    _FALLBACK_STORE.clear()


@pytest.fixture
def online_store() -> OnlineFeatureStore:
    return OnlineFeatureStore(redis_url="redis://127.0.0.1:16379", default_ttl=3600)


@pytest.fixture
def offline_store(tmp_path) -> OfflineFeatureStore:
    return OfflineFeatureStore(base_path=str(tmp_path))


@pytest.fixture
def checker(online_store, offline_store) -> ConsistencyChecker:
    return ConsistencyChecker(offline_store=offline_store, online_store=online_store)


def populate_consistent(online_store, offline_store, entity_ids, feature_name, value, ts):
    """Write the same value to both online and offline stores for all entity_ids."""
    for eid in entity_ids:
        online_store.write_user_features(eid, {feature_name: value})
        offline_store.write("user", eid, {feature_name: value}, ts)


# ---------------------------------------------------------------------------
# Basic consistency tests
# ---------------------------------------------------------------------------


class TestConsistentData:
    def test_consistent_returns_passed_true(self, checker, online_store, offline_store):
        entity_ids = [1, 2, 3]
        ts = BASE - timedelta(hours=1)
        populate_consistent(online_store, offline_store, entity_ids, "purchase_count_24h", 5, ts)

        report = checker.check("user", entity_ids, ["purchase_count_24h"], as_of=BASE)
        assert report.passed is True
        assert "purchase_count_24h" not in report.inconsistent_features

    def test_consistent_multiple_features(self, checker, online_store, offline_store):
        entity_ids = [10, 11, 12]
        ts = BASE - timedelta(minutes=30)
        features = {"purchase_count_24h": 3, "total_spend_24h": 99.99}
        for eid in entity_ids:
            online_store.write_user_features(eid, features)
            offline_store.write("user", eid, features, ts)

        report = checker.check("user", entity_ids, list(features.keys()), as_of=BASE)
        assert report.passed is True
        assert report.inconsistent_features == []

    def test_report_contains_per_feature_stats(self, checker, online_store, offline_store):
        entity_ids = [1, 2]
        ts = BASE - timedelta(hours=1)
        populate_consistent(online_store, offline_store, entity_ids, "purchase_count_24h", 7, ts)

        report = checker.check("user", entity_ids, ["purchase_count_24h"], as_of=BASE)
        assert len(report.per_feature_stats) == 1
        stats = report.per_feature_stats[0]
        assert stats.feature_name == "purchase_count_24h"
        assert stats.n_compared >= 1
        assert stats.inconsistency_rate == 0.0


# ---------------------------------------------------------------------------
# Inconsistency detection
# ---------------------------------------------------------------------------


class TestInconsistencyDetection:
    def test_large_divergence_is_detected(self, checker, online_store, offline_store):
        """Online value is 100x the offline value — clear inconsistency."""
        entity_ids = list(range(1, 11))  # 10 entities
        ts = BASE - timedelta(hours=1)

        for eid in entity_ids:
            offline_store.write("user", eid, {"total_spend_24h": 100.0}, ts)
            # Online value is wildly different
            online_store.write_user_features(eid, {"total_spend_24h": 99999.0})

        report = checker.check("user", entity_ids, ["total_spend_24h"], as_of=BASE)
        assert report.passed is False
        assert "total_spend_24h" in report.inconsistent_features

    def test_small_divergence_below_threshold_is_not_flagged(self, checker, online_store, offline_store):
        """A <5% difference in a few entities should not trigger the flag."""
        entity_ids = list(range(1, 21))  # 20 entities
        ts = BASE - timedelta(hours=1)

        for eid in entity_ids:
            offline_val = 100.0
            # Introduce <1% difference for a few entities — well below threshold
            online_val = offline_val * 1.001 if eid % 5 == 0 else offline_val
            offline_store.write("user", eid, {"avg_session_duration": offline_val}, ts)
            online_store.write_user_features(eid, {"avg_session_duration": online_val})

        report = checker.check("user", entity_ids, ["avg_session_duration"], as_of=BASE)
        assert report.passed is True

    def test_categorical_inconsistency_detected(self, checker, online_store, offline_store):
        """Mismatched category_affinity lists should be detected."""
        entity_ids = list(range(1, 11))
        ts = BASE - timedelta(hours=1)

        for eid in entity_ids:
            offline_store.write("user", eid, {"category_affinity": '["electronics", "clothing"]'}, ts)
            # Serve a completely different affinity from online store
            online_store.write_user_features(eid, {"category_affinity": ["sports", "books"]})

        report = checker.check("user", entity_ids, ["category_affinity"], as_of=BASE)
        assert report.passed is False
        assert "category_affinity" in report.inconsistent_features

    def test_per_feature_stats_accuracy(self, checker, online_store, offline_store):
        """Stats should correctly identify fraction of inconsistent entities."""
        entity_ids = list(range(1, 11))  # 10 entities
        ts = BASE - timedelta(hours=1)
        n_inconsistent = 8  # 80% — above threshold

        for eid in entity_ids:
            offline_val = 50.0
            if eid <= n_inconsistent:
                # Large divergence (200%) for n_inconsistent entities
                online_val = 150.0
            else:
                online_val = 50.0
            offline_store.write("user", eid, {"purchase_count_24h": offline_val}, ts)
            online_store.write_user_features(eid, {"purchase_count_24h": online_val})

        report = checker.check("user", entity_ids, ["purchase_count_24h"], as_of=BASE)
        stats = next(s for s in report.per_feature_stats if s.feature_name == "purchase_count_24h")

        assert stats.n_inconsistent == n_inconsistent
        assert abs(stats.inconsistency_rate - n_inconsistent / len(entity_ids)) < 0.01
        assert stats.flagged is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_entity_list_returns_empty_report(self, checker):
        report = checker.check("user", [], ["purchase_count_24h"])
        assert report.total_checked == 0
        assert report.passed is True

    def test_no_offline_data_skips_entities(self, checker, online_store):
        """Entities with no offline snapshot are skipped (can't compare)."""
        online_store.write_user_features(1, {"purchase_count_24h": 5})
        # No offline write — nothing to compare against
        report = checker.check("user", [1], ["purchase_count_24h"])
        # No pairs means no flagging
        assert report.passed is True

    def test_report_summary_contains_entity(self, checker, online_store, offline_store):
        entity_ids = [1]
        ts = BASE - timedelta(hours=1)
        populate_consistent(online_store, offline_store, entity_ids, "purchase_count_24h", 1, ts)

        report = checker.check("user", entity_ids, ["purchase_count_24h"], as_of=BASE)
        summary = report.summary()
        assert "user" in summary
        assert "PASSED" in summary

    def test_item_entity_type(self, checker, online_store, offline_store):
        ts = BASE - timedelta(hours=1)
        offline_store.write("item", 100, {"view_count_24h": 50}, ts)
        online_store.write_item_features(100, {"view_count_24h": 50})

        report = checker.check("item", [100], ["view_count_24h"], as_of=BASE)
        assert report.passed is True
