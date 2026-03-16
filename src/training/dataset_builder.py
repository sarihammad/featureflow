"""Point-in-time correct training dataset builder.

The core problem this module solves:

    Naive approach: join label events to the *latest* feature values for each
    entity.  This causes data leakage because features computed *after* the
    label event timestamp contaminate the training row — the model sees the
    future.  Training metrics look great; production performance is terrible.

    Correct approach: for each label event at time T, find each entity's
    feature snapshot with the largest timestamp that is strictly less than T.
    This is a "point-in-time join" or "as-of join".
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.features.registry import FeatureRegistry
from src.stores.offline_store import TIMESTAMP_COL, OfflineFeatureStore

logger = logging.getLogger(__name__)

LABEL_TIMESTAMP_COL = "label_timestamp"
FEATURE_TIMESTAMP_COL = "feature_timestamp_used"


@dataclass
class LeakageReport:
    passed: bool
    violations: int
    total_rows: int
    details: List[str]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"LeakageReport [{status}] "
            f"{self.violations}/{self.total_rows} violations"
        )


class PointInTimeDatasetBuilder:
    """Assembles a training dataset free of label leakage.

    Usage::

        builder = PointInTimeDatasetBuilder(offline_store, registry)
        df = builder.build(
            label_events=labels_df,          # user_id, item_id, timestamp, label
            user_features=["purchase_count_24h", "total_spend_24h"],
            item_features=["view_count_24h", "avg_rating"],
        )
    """

    def __init__(
        self,
        offline_store: OfflineFeatureStore,
        registry: FeatureRegistry,
    ):
        self._offline = offline_store
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        label_events: pd.DataFrame,
        user_features: List[str],
        item_features: List[str],
    ) -> pd.DataFrame:
        """Return a point-in-time correct training DataFrame.

        Parameters
        ----------
        label_events:
            DataFrame with at minimum columns: ``user_id``, ``timestamp``,
            ``label``.  Optionally ``item_id`` for item-side features.
        user_features:
            List of user feature names to attach.
        item_features:
            List of item feature names to attach.  Ignored if ``item_id``
            column is absent from ``label_events``.

        Returns
        -------
        pd.DataFrame
            One row per label event that had a prior feature snapshot.
            Rows with no prior history (cold start) are dropped.
        """
        required_cols = {"user_id", "timestamp", "label"}
        missing = required_cols - set(label_events.columns)
        if missing:
            raise ValueError(f"label_events is missing required columns: {missing}")

        label_events = label_events.copy()
        label_events["timestamp"] = pd.to_datetime(label_events["timestamp"], utc=True)

        result_rows: List[Dict[str, Any]] = []
        has_item_id = "item_id" in label_events.columns

        for _, row in label_events.iterrows():
            user_id = int(row["user_id"])
            label_ts: datetime = row["timestamp"].to_pydatetime()
            if label_ts.tzinfo is None:
                label_ts = label_ts.replace(tzinfo=timezone.utc)

            user_feat = self._get_features_as_of("user", user_id, label_ts, user_features)
            if user_feat is None:
                logger.debug(
                    "No prior user feature history for user_id=%d at %s — dropping row",
                    user_id,
                    label_ts.isoformat(),
                )
                continue

            item_feat: Dict[str, Any] = {}
            feature_ts_used = user_feat.pop(FEATURE_TIMESTAMP_COL)

            if has_item_id and item_features:
                item_id = row.get("item_id")
                if item_id is not None and not pd.isna(item_id):
                    item_id = int(item_id)
                    item_data = self._get_features_as_of("item", item_id, label_ts, item_features)
                    if item_data is not None:
                        item_data.pop(FEATURE_TIMESTAMP_COL, None)
                        item_feat = {f"item_{k}": v for k, v in item_data.items()}

            output_row: Dict[str, Any] = {
                "user_id": user_id,
                "label": row["label"],
                LABEL_TIMESTAMP_COL: label_ts,
                FEATURE_TIMESTAMP_COL: feature_ts_used,
            }
            if has_item_id:
                output_row["item_id"] = row.get("item_id")

            output_row.update({f"user_{k}": v for k, v in user_feat.items()})
            output_row.update(item_feat)

            result_rows.append(output_row)

        if not result_rows:
            logger.warning("No rows survived the point-in-time join (all cold start).")
            return pd.DataFrame()

        dataset = pd.DataFrame(result_rows)
        logger.info(
            "PIT join complete: %d label events → %d training rows (%.1f%% kept)",
            len(label_events),
            len(dataset),
            100 * len(dataset) / len(label_events),
        )
        return dataset

    def _get_features_as_of(
        self,
        entity: str,
        entity_id: int,
        timestamp: datetime,
        feature_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Return the entity's feature values strictly before *timestamp*.

        Returns ``None`` when no prior snapshot exists (cold start).
        """
        from datetime import timedelta

        # Read a generous window of history for this entity (up to 30 days back)
        start_time = timestamp - timedelta(days=30)
        history = self._offline.read_entity_history(
            entity=entity,
            entity_id=entity_id,
            start_time=start_time,
            end_time=timestamp,  # exclusive upper bound
        )

        if history.empty:
            return None

        # Take the most recent snapshot strictly before timestamp
        history_sorted = history.sort_values(TIMESTAMP_COL)
        latest = history_sorted.iloc[-1]

        result: Dict[str, Any] = {}
        for fname in feature_names:
            if fname in latest.index:
                val = latest[fname]
                # Attempt to deserialise list/dict values stored as JSON strings
                if isinstance(val, str) and val.startswith(("[", "{")):
                    import json
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                result[fname] = val
            else:
                result[fname] = None

        result[FEATURE_TIMESTAMP_COL] = latest[TIMESTAMP_COL]
        return result

    # ------------------------------------------------------------------
    # Leakage validation
    # ------------------------------------------------------------------

    def validate_no_leakage(
        self,
        dataset: pd.DataFrame,
        label_events: pd.DataFrame,
    ) -> LeakageReport:
        """Verify that every feature snapshot predates its label event.

        A violation occurs when ``feature_timestamp_used >= label_timestamp``,
        which means a future feature value was used during training.
        """
        if FEATURE_TIMESTAMP_COL not in dataset.columns:
            return LeakageReport(
                passed=False,
                violations=len(dataset),
                total_rows=len(dataset),
                details=["Column 'feature_timestamp_used' missing from dataset."],
            )
        if LABEL_TIMESTAMP_COL not in dataset.columns:
            return LeakageReport(
                passed=False,
                violations=len(dataset),
                total_rows=len(dataset),
                details=["Column 'label_timestamp' missing from dataset."],
            )

        feat_ts = pd.to_datetime(dataset[FEATURE_TIMESTAMP_COL], utc=True)
        label_ts = pd.to_datetime(dataset[LABEL_TIMESTAMP_COL], utc=True)

        violation_mask = feat_ts >= label_ts
        violations = int(violation_mask.sum())

        details: List[str] = []
        for idx in dataset[violation_mask].index[:10]:  # cap detail output at 10
            details.append(
                f"Row {idx}: feature_ts={feat_ts[idx].isoformat()} "
                f">= label_ts={label_ts[idx].isoformat()}"
            )

        return LeakageReport(
            passed=violations == 0,
            violations=violations,
            total_rows=len(dataset),
            details=details,
        )
