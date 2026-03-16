"""Tests for point-in-time correct training dataset generation.

These tests verify the core correctness guarantee: features attached to each
training row must have been computed *before* the label event timestamp.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.features.registry import FeatureRegistry, _registry
from src.stores.offline_store import TIMESTAMP_COL, FeatureRecord, OfflineFeatureStore
from src.training.dataset_builder import (
    FEATURE_TIMESTAMP_COL,
    LABEL_TIMESTAMP_COL,
    LeakageReport,
    PointInTimeDatasetBuilder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def offline_store(tmp_path) -> OfflineFeatureStore:
    return OfflineFeatureStore(base_path=str(tmp_path))


@pytest.fixture
def registry() -> FeatureRegistry:
    return _registry


@pytest.fixture
def builder(offline_store, registry) -> PointInTimeDatasetBuilder:
    return PointInTimeDatasetBuilder(offline_store=offline_store, registry=registry)


def write_user_snapshot(
    store: OfflineFeatureStore,
    user_id: int,
    features: dict,
    ts: datetime,
) -> None:
    store.write("user", user_id, features, ts)


def make_label_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "label"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Core PIT correctness
# ---------------------------------------------------------------------------


class TestPointInTimeCorrectness:
    def test_uses_features_strictly_before_label_timestamp(self, builder, offline_store):
        """The snapshot at T-1h should be used, not the one at T+1h."""
        user_id = 1

        # Snapshot BEFORE label event
        ts_before = BASE - timedelta(hours=1)
        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 3}, ts_before)

        # Snapshot AFTER label event — must NOT be used
        ts_after = BASE + timedelta(hours=1)
        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 99}, ts_after)

        labels = make_label_df([
            (user_id, None, BASE, 1),
        ])

        dataset = builder.build(labels, user_features=["purchase_count_24h"], item_features=[])

        assert not dataset.empty
        row = dataset.iloc[0]
        # Should have used the snapshot from ts_before, not ts_after
        assert row["user_purchase_count_24h"] == 3

    def test_uses_most_recent_snapshot_before_label(self, builder, offline_store):
        user_id = 2

        ts_old = BASE - timedelta(hours=5)
        ts_new = BASE - timedelta(hours=1)
        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 1}, ts_old)
        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 7}, ts_new)

        labels = make_label_df([(user_id, None, BASE, 0)])
        dataset = builder.build(labels, user_features=["purchase_count_24h"], item_features=[])

        assert dataset.iloc[0]["user_purchase_count_24h"] == 7

    def test_different_label_timestamps_get_different_snapshots(self, builder, offline_store):
        """Two label events for the same user at different times should get
        the snapshot that was latest *before each label timestamp*."""
        user_id = 3

        ts_snap1 = BASE - timedelta(hours=6)
        ts_snap2 = BASE - timedelta(hours=2)
        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 5}, ts_snap1)
        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 20}, ts_snap2)

        label_t1 = BASE - timedelta(hours=3)  # only snap1 is before this
        label_t2 = BASE                        # snap2 is also before this

        labels = make_label_df([
            (user_id, None, label_t1, 1),
            (user_id, None, label_t2, 0),
        ])
        dataset = builder.build(labels, user_features=["purchase_count_24h"], item_features=[])

        assert len(dataset) == 2
        # Sort by label timestamp to make assertions deterministic
        dataset = dataset.sort_values(LABEL_TIMESTAMP_COL).reset_index(drop=True)
        assert dataset.iloc[0]["user_purchase_count_24h"] == 5   # label at t1 gets snap1
        assert dataset.iloc[1]["user_purchase_count_24h"] == 20  # label at t2 gets snap2


# ---------------------------------------------------------------------------
# Cold start handling
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_rows_with_no_prior_history_are_dropped(self, builder):
        """An entity with no feature snapshots before the label timestamp
        should be silently dropped from the result."""
        labels = make_label_df([
            (9999, None, BASE, 1),  # no snapshots exist for this user
        ])
        dataset = builder.build(labels, user_features=["purchase_count_24h"], item_features=[])
        assert dataset.empty

    def test_partial_cold_start_keeps_warm_rows(self, builder, offline_store):
        user_warm = 10
        user_cold = 11

        write_user_snapshot(offline_store, user_warm, {"purchase_count_24h": 3}, BASE - timedelta(hours=2))

        labels = make_label_df([
            (user_warm, None, BASE, 1),
            (user_cold, None, BASE, 0),  # cold — no history
        ])
        dataset = builder.build(labels, user_features=["purchase_count_24h"], item_features=[])

        assert len(dataset) == 1
        assert int(dataset.iloc[0]["user_id"]) == user_warm


# ---------------------------------------------------------------------------
# Item feature joining
# ---------------------------------------------------------------------------


class TestItemFeatureJoin:
    def test_item_features_attached_with_prefix(self, builder, offline_store):
        user_id, item_id = 20, 200

        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 2}, BASE - timedelta(hours=1))
        offline_store.write(
            "item", item_id, {"view_count_24h": 150, "avg_rating": 4.1},
            BASE - timedelta(hours=1),
        )

        labels = make_label_df([(user_id, item_id, BASE, 1)])
        dataset = builder.build(
            labels,
            user_features=["purchase_count_24h"],
            item_features=["view_count_24h", "avg_rating"],
        )

        assert not dataset.empty
        row = dataset.iloc[0]
        assert "item_view_count_24h" in row
        assert "item_avg_rating" in row
        assert row["item_view_count_24h"] == 150

    def test_missing_item_history_does_not_drop_row(self, builder, offline_store):
        """If item features are unavailable but user features exist,
        the row should still appear (item features will be None)."""
        user_id, item_id = 30, 300

        write_user_snapshot(offline_store, user_id, {"purchase_count_24h": 1}, BASE - timedelta(hours=1))
        # No item snapshot written

        labels = make_label_df([(user_id, item_id, BASE, 1)])
        dataset = builder.build(
            labels,
            user_features=["purchase_count_24h"],
            item_features=["view_count_24h"],
        )

        assert len(dataset) == 1


# ---------------------------------------------------------------------------
# Leakage validation
# ---------------------------------------------------------------------------


class TestLeakageValidation:
    def _clean_dataset(self) -> pd.DataFrame:
        rows = []
        for i in range(5):
            label_ts = BASE + timedelta(hours=i)
            feature_ts = label_ts - timedelta(hours=1)
            rows.append({
                "user_id": i,
                LABEL_TIMESTAMP_COL: label_ts,
                FEATURE_TIMESTAMP_COL: feature_ts,
                "label": 1,
            })
        return pd.DataFrame(rows)

    def _leaky_dataset(self) -> pd.DataFrame:
        df = self._clean_dataset()
        # Introduce a violation: feature_ts >= label_ts
        df.at[2, FEATURE_TIMESTAMP_COL] = df.at[2, LABEL_TIMESTAMP_COL] + timedelta(seconds=1)
        return df

    def test_validate_passes_on_clean_data(self, builder):
        dataset = self._clean_dataset()
        report = builder.validate_no_leakage(dataset, dataset)
        assert report.passed is True
        assert report.violations == 0

    def test_validate_detects_injected_leakage(self, builder):
        dataset = self._leaky_dataset()
        report = builder.validate_no_leakage(dataset, dataset)
        assert report.passed is False
        assert report.violations >= 1

    def test_validate_all_leaky(self, builder):
        rows = []
        for i in range(3):
            label_ts = BASE + timedelta(hours=i)
            # feature_ts == label_ts — violation
            rows.append({
                "user_id": i,
                LABEL_TIMESTAMP_COL: label_ts,
                FEATURE_TIMESTAMP_COL: label_ts,  # equal, not strictly before
                "label": 1,
            })
        dataset = pd.DataFrame(rows)
        report = builder.validate_no_leakage(dataset, dataset)
        assert report.passed is False
        assert report.violations == 3

    def test_violation_details_populated(self, builder):
        dataset = self._leaky_dataset()
        report = builder.validate_no_leakage(dataset, dataset)
        assert len(report.details) >= 1
        assert "feature_ts" in report.details[0]

    def test_leakage_report_str_representation(self, builder):
        dataset = self._clean_dataset()
        report = builder.validate_no_leakage(dataset, dataset)
        s = str(report)
        assert "PASSED" in s

    def test_validate_missing_column_returns_failed(self, builder):
        df = pd.DataFrame({"user_id": [1], "label": [1]})
        report = builder.validate_no_leakage(df, df)
        assert report.passed is False
