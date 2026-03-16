"""Tests for online and offline feature stores.

Online store tests run against the in-memory fallback so no Redis is required.
Offline store tests write to a temporary directory.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from src.stores.offline_store import (
    ENTITY_ID_COL,
    TIMESTAMP_COL,
    FeatureRecord,
    OfflineFeatureStore,
)
from src.stores.online_store import OnlineFeatureStore, _FALLBACK_STORE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_fallback_store():
    """Ensure the global in-memory fallback is empty before each test."""
    _FALLBACK_STORE.clear()
    yield
    _FALLBACK_STORE.clear()


@pytest.fixture
def online_store() -> OnlineFeatureStore:
    # Force fallback mode by pointing to a non-existent Redis instance
    store = OnlineFeatureStore(redis_url="redis://127.0.0.1:16379", default_ttl=3600)
    assert store._fallback_active, "Test expects fallback mode"
    return store


@pytest.fixture
def offline_store(tmp_path) -> OfflineFeatureStore:
    return OfflineFeatureStore(base_path=str(tmp_path))


# ---------------------------------------------------------------------------
# Online store tests
# ---------------------------------------------------------------------------


class TestOnlineStoreWriteRead:
    def test_user_features_round_trip(self, online_store):
        features = {
            "purchase_count_24h": 5,
            "total_spend_24h": 149.95,
            "category_affinity": ["electronics", "clothing"],
        }
        online_store.write_user_features(user_id=42, features=features)
        result = online_store.read_user_features(42, list(features.keys()))
        assert result["purchase_count_24h"] == 5
        assert abs(result["total_spend_24h"] - 149.95) < 1e-6
        assert result["category_affinity"] == ["electronics", "clothing"]

    def test_item_features_round_trip(self, online_store):
        features = {"view_count_1h": 100, "avg_rating": 4.2}
        online_store.write_item_features(item_id=7, features=features)
        result = online_store.read_item_features(7, list(features.keys()))
        assert result["view_count_1h"] == 100
        assert result["avg_rating"] == pytest.approx(4.2)

    def test_missing_feature_returns_none(self, online_store):
        online_store.write_user_features(1, {"purchase_count_24h": 3})
        result = online_store.read_user_features(1, ["purchase_count_24h", "nonexistent"])
        assert result["purchase_count_24h"] == 3
        assert result["nonexistent"] is None

    def test_feature_vector_merges_user_and_item(self, online_store):
        online_store.write_user_features(10, {"purchase_count_24h": 2})
        online_store.write_item_features(20, {"view_count_1h": 50})

        vector = online_store.read_feature_vector(
            user_id=10,
            item_id=20,
            user_feature_names=["purchase_count_24h"],
            item_feature_names=["view_count_1h"],
        )
        assert vector["user_purchase_count_24h"] == 2
        assert vector["item_view_count_1h"] == 50

    def test_feature_vector_prefixes_correctly(self, online_store):
        online_store.write_user_features(1, {"total_spend_24h": 99.0})
        online_store.write_item_features(1, {"revenue_24h": 500.0})
        # Both entities have id=1 but different entity types
        vector = online_store.read_feature_vector(
            user_id=1,
            item_id=1,
            user_feature_names=["total_spend_24h"],
            item_feature_names=["revenue_24h"],
        )
        assert "user_total_spend_24h" in vector
        assert "item_revenue_24h" in vector
        assert vector["user_total_spend_24h"] == pytest.approx(99.0)
        assert vector["item_revenue_24h"] == pytest.approx(500.0)

    def test_overwrite_updates_value(self, online_store):
        online_store.write_user_features(5, {"purchase_count_1h": 1})
        online_store.write_user_features(5, {"purchase_count_1h": 10})
        result = online_store.read_user_features(5, ["purchase_count_1h"])
        assert result["purchase_count_1h"] == 10

    def test_delete_user_features(self, online_store):
        online_store.write_user_features(99, {"purchase_count_24h": 7})
        online_store.delete_user_features(99)
        result = online_store.read_user_features(99, ["purchase_count_24h"])
        assert result["purchase_count_24h"] is None

    def test_health_check_returns_true_in_fallback(self, online_store):
        assert online_store.health_check() is True

    def test_json_serialisation_of_list_values(self, online_store):
        online_store.write_user_features(3, {"category_affinity": ["a", "b", "c"]})
        result = online_store.read_user_features(3, ["category_affinity"])
        assert result["category_affinity"] == ["a", "b", "c"]

    def test_float_precision(self, online_store):
        online_store.write_user_features(1, {"conversion_rate_7d": 0.123456789})
        result = online_store.read_user_features(1, ["conversion_rate_7d"])
        assert result["conversion_rate_7d"] == pytest.approx(0.123456789, rel=1e-6)


# ---------------------------------------------------------------------------
# Offline store tests
# ---------------------------------------------------------------------------


class TestOfflineStoreWriteRead:
    def test_write_and_read_single_record(self, offline_store):
        ts = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        offline_store.write(
            entity="user",
            entity_id=42,
            features={"purchase_count_24h": 5, "total_spend_24h": 99.0},
            timestamp=ts,
        )
        history = offline_store.read_entity_history(
            entity="user",
            entity_id=42,
            start_time=ts - timedelta(hours=1),
            end_time=ts + timedelta(hours=1),
        )
        assert not history.empty
        assert int(history.iloc[0][ENTITY_ID_COL]) == 42

    def test_read_returns_only_matching_entity(self, offline_store):
        ts = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        offline_store.write("user", 1, {"purchase_count_24h": 1}, ts)
        offline_store.write("user", 2, {"purchase_count_24h": 2}, ts)

        history = offline_store.read_entity_history(
            "user", 1,
            start_time=ts - timedelta(hours=1),
            end_time=ts + timedelta(hours=1),
        )
        assert all(history[ENTITY_ID_COL] == 1)

    def test_read_entity_history_respects_time_range(self, offline_store):
        ts1 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts3 = datetime(2024, 6, 1, 14, 0, 0, tzinfo=timezone.utc)

        for ts, val in [(ts1, 1), (ts2, 2), (ts3, 3)]:
            offline_store.write("user", 1, {"purchase_count_24h": val}, ts)

        history = offline_store.read_entity_history(
            "user", 1,
            start_time=ts1,
            end_time=ts3,  # exclusive — ts3 should not appear
        )
        assert len(history) == 2
        values = sorted(history["purchase_count_24h"].tolist())
        assert values == [1, 2]

    def test_write_batch(self, offline_store):
        ts = datetime(2024, 6, 2, 8, 0, 0, tzinfo=timezone.utc)
        records = [
            FeatureRecord("user", uid, {"purchase_count_24h": uid * 10}, ts)
            for uid in range(1, 6)
        ]
        offline_store.write_batch(records)

        for uid in range(1, 6):
            history = offline_store.read_entity_history(
                "user", uid,
                start_time=ts - timedelta(minutes=1),
                end_time=ts + timedelta(minutes=1),
            )
            assert not history.empty

    def test_read_all_latest_returns_one_row_per_entity(self, offline_store):
        ts1 = datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

        for uid in [1, 2]:
            offline_store.write("user", uid, {"purchase_count_24h": 1}, ts1)
            offline_store.write("user", uid, {"purchase_count_24h": 5}, ts2)

        as_of = ts2 + timedelta(minutes=1)
        latest = offline_store.read_all_latest("user", as_of)

        assert len(latest) == 2
        for _, row in latest.iterrows():
            assert row["purchase_count_24h"] == 5

    def test_read_all_latest_excludes_future_snapshots(self, offline_store):
        ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        offline_store.write("user", 1, {"purchase_count_24h": 10}, ts)

        # as_of is before the snapshot
        as_of = ts - timedelta(minutes=1)
        latest = offline_store.read_all_latest("user", as_of)
        assert latest.empty

    def test_list_serialisation_preserved(self, offline_store):
        ts = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        offline_store.write("user", 1, {"category_affinity": ["electronics", "books"]}, ts)

        history = offline_store.read_entity_history(
            "user", 1,
            start_time=ts - timedelta(hours=1),
            end_time=ts + timedelta(hours=1),
        )
        raw_val = history.iloc[0]["category_affinity"]
        # Should be deserialised back to list or stored as JSON string
        if isinstance(raw_val, str):
            parsed = json.loads(raw_val)
        else:
            parsed = raw_val
        assert parsed == ["electronics", "books"]

    def test_empty_store_returns_empty_df(self, offline_store):
        ts = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        result = offline_store.read_entity_history(
            "user", 9999,
            start_time=ts - timedelta(hours=1),
            end_time=ts + timedelta(hours=1),
        )
        assert result.empty
