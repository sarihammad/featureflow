"""Training-serving consistency checker.

Training-serving skew is one of the most insidious production ML bugs: the
model was trained on features computed one way, but at serving time the same
feature name refers to a value computed differently (different window, different
preprocessing, stale cache, code divergence).  Skew is silent — the model just
performs worse than expected with no obvious error.

This module compares online (Redis) feature values to the most recent offline
(Parquet) feature values for a sample of entities and reports which features
are inconsistent and by how much.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.stores.offline_store import OfflineFeatureStore
from src.stores.online_store import OnlineFeatureStore

logger = logging.getLogger(__name__)

# A feature is considered inconsistent when the relative difference between
# online and offline values exceeds this threshold for more than
# INCONSISTENCY_RATE_THRESHOLD fraction of entities.
RELATIVE_DIFF_THRESHOLD = 0.05       # 5% relative difference
INCONSISTENCY_RATE_THRESHOLD = 0.05  # flag if >5% of entities are inconsistent


@dataclass
class FeatureStats:
    feature_name: str
    entity_type: str
    n_compared: int
    n_inconsistent: int
    inconsistency_rate: float
    mean_relative_diff: float
    max_relative_diff: float
    flagged: bool


@dataclass
class ConsistencyReport:
    entity: str
    total_checked: int
    inconsistent_features: List[str]
    per_feature_stats: List[FeatureStats]
    passed: bool
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        flagged = ", ".join(self.inconsistent_features) or "none"
        return (
            f"ConsistencyReport [{status}] entity={self.entity} "
            f"checked={self.total_checked} flagged_features=[{flagged}]"
        )


class ConsistencyChecker:
    """Compare online (Redis) vs offline (Parquet) feature values.

    For numerical features the comparison metric is relative difference::

        |online - offline| / max(|offline|, 1e-9)

    For list/categorical features exact equality is used.
    """

    def __init__(
        self,
        offline_store: OfflineFeatureStore,
        online_store: OnlineFeatureStore,
    ):
        self._offline = offline_store
        self._online = online_store

    def check(
        self,
        entity: str,
        entity_ids: List[int],
        feature_names: List[str],
        as_of: Optional[datetime] = None,
    ) -> ConsistencyReport:
        """Run consistency check for *entity_ids* and *feature_names*.

        Parameters
        ----------
        entity:
            ``"user"`` or ``"item"``
        entity_ids:
            Sample of entity IDs to check.
        feature_names:
            Which features to compare.
        as_of:
            Use the offline snapshot as-of this timestamp (defaults to now).
        """
        if as_of is None:
            as_of = datetime.utcnow()

        # Fetch the latest offline snapshot for all entities at once
        offline_latest = self._offline.read_all_latest(entity=entity, as_of=as_of)

        # Build a lookup: entity_id -> {feature_name: value}
        offline_lookup: Dict[int, Dict[str, Any]] = {}
        if not offline_latest.empty and "entity_id" in offline_latest.columns:
            for _, row in offline_latest.iterrows():
                eid = int(row["entity_id"])
                offline_lookup[eid] = {
                    fname: row.get(fname) for fname in feature_names if fname in row.index
                }

        # Per-feature lists of (online_val, offline_val) tuples
        comparisons: Dict[str, List[Tuple[Any, Any]]] = {f: [] for f in feature_names}

        for entity_id in entity_ids:
            # Fetch online features
            if entity == "user":
                online_vals = self._online.read_user_features(entity_id, feature_names)
            else:
                online_vals = self._online.read_item_features(entity_id, feature_names)

            offline_vals = offline_lookup.get(entity_id, {})

            for fname in feature_names:
                online_v = online_vals.get(fname)
                offline_v = offline_vals.get(fname)
                if online_v is None or offline_v is None:
                    continue  # can't compare if one side is missing
                comparisons[fname].append((online_v, offline_v))

        # Compute per-feature statistics
        per_feature_stats: List[FeatureStats] = []
        inconsistent_features: List[str] = []

        for fname in feature_names:
            pairs = comparisons[fname]
            if not pairs:
                continue

            diffs: List[float] = []
            n_inconsistent = 0

            for online_v, offline_v in pairs:
                diff, is_inconsistent = self._compute_diff(online_v, offline_v)
                diffs.append(diff)
                if is_inconsistent:
                    n_inconsistent += 1

            n = len(pairs)
            rate = n_inconsistent / n if n > 0 else 0.0
            mean_diff = float(np.mean(diffs)) if diffs else 0.0
            max_diff = float(np.max(diffs)) if diffs else 0.0
            flagged = rate > INCONSISTENCY_RATE_THRESHOLD

            stats = FeatureStats(
                feature_name=fname,
                entity_type=entity,
                n_compared=n,
                n_inconsistent=n_inconsistent,
                inconsistency_rate=round(rate, 4),
                mean_relative_diff=round(mean_diff, 4),
                max_relative_diff=round(max_diff, 4),
                flagged=flagged,
            )
            per_feature_stats.append(stats)

            if flagged:
                inconsistent_features.append(fname)
                logger.warning(
                    "Feature '%s' inconsistency: %.1f%% of %d entities differ "
                    "(mean_rel_diff=%.4f)",
                    fname,
                    100 * rate,
                    n,
                    mean_diff,
                )

        report = ConsistencyReport(
            entity=entity,
            total_checked=len(entity_ids),
            inconsistent_features=inconsistent_features,
            per_feature_stats=per_feature_stats,
            passed=len(inconsistent_features) == 0,
            checked_at=as_of,
        )
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diff(online_v: Any, offline_v: Any) -> Tuple[float, bool]:
        """Return (relative_diff, is_inconsistent)."""
        # List / categorical: exact match
        if isinstance(online_v, (list, str)) or isinstance(offline_v, (list, str)):
            equal = online_v == offline_v
            return (0.0 if equal else 1.0, not equal)

        # Numerical: relative difference
        try:
            a = float(online_v)
            b = float(offline_v)
        except (TypeError, ValueError):
            equal = online_v == offline_v
            return (0.0 if equal else 1.0, not equal)

        denom = max(abs(b), 1e-9)
        rel_diff = abs(a - b) / denom
        return (rel_diff, rel_diff > RELATIVE_DIFF_THRESHOLD)
