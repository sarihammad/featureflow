"""Batch processor for historical feature backfill.

This component answers the question: "Given a dump of historical events, what
were each entity's features at hourly intervals over a date range?"  The
resulting Parquet snapshots are what the PointInTimeDatasetBuilder reads when
assembling training data.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from src.events.schema import UserEvent
from src.features.registry import FeatureRegistry
from src.features import transformations as T
from src.stores.offline_store import FeatureRecord, OfflineFeatureStore

logger = logging.getLogger(__name__)


class BatchProcessor:
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

    def backfill_from_events(
        self,
        events: List[UserEvent],
        as_of_times: List[datetime],
    ) -> None:
        """Compute and persist feature snapshots at each as_of_time.

        For each timestamp, only events strictly before that timestamp are
        used, preserving point-in-time correctness.
        """
        if not events or not as_of_times:
            return

        # Pre-sort events once
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for as_of in sorted(as_of_times):
            eligible = [e for e in sorted_events if e.timestamp < as_of]
            if not eligible:
                logger.debug("No events before %s — skipping snapshot", as_of.isoformat())
                continue

            self._compute_and_write_snapshot(eligible, as_of)
            logger.info("Backfilled snapshot at %s (%d events)", as_of.isoformat(), len(eligible))

    def materialize_date_range(
        self,
        events: List[UserEvent],
        start_date: datetime,
        end_date: datetime,
        freq: str = "1H",
    ) -> None:
        """Generate feature snapshots at regular intervals over [start_date, end_date].

        ``freq`` supports ``"1H"`` (hourly), ``"6H"``, ``"1D"``, etc.
        """
        freq_map = {
            "1H": timedelta(hours=1),
            "6H": timedelta(hours=6),
            "12H": timedelta(hours=12),
            "1D": timedelta(days=1),
        }
        delta = freq_map.get(freq)
        if delta is None:
            raise ValueError(f"Unsupported freq={freq!r}. Choose from {list(freq_map)}")

        as_of_times: List[datetime] = []
        current = start_date
        while current <= end_date:
            as_of_times.append(current)
            current += delta

        self.backfill_from_events(events, as_of_times)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_and_write_snapshot(
        self,
        eligible_events: List[UserEvent],
        ref_time: datetime,
    ) -> None:
        user_events: Dict[int, List[UserEvent]] = defaultdict(list)
        item_events: Dict[int, List[UserEvent]] = defaultdict(list)

        for e in eligible_events:
            user_events[e.user_id].append(e)
            if e.item_id is not None:
                item_events[e.item_id].append(e)

        # Compute relative popularity ranks for items
        item_view_counts: Dict[int, int] = {}
        for item_id, evts in item_events.items():
            item_view_counts[item_id] = T.item_view_count_for_item(evts, item_id, 3600, ref_time)
        max_views = max(item_view_counts.values(), default=1)

        records: List[FeatureRecord] = []

        for user_id, evts in user_events.items():
            features = {
                "purchase_count_1h": T.purchase_count(evts, 3600, ref_time),
                "purchase_count_24h": T.purchase_count(evts, 86400, ref_time),
                "item_view_count_1h": T.item_view_count(evts, 3600, ref_time),
                "item_view_count_24h": T.item_view_count(evts, 86400, ref_time),
                "cart_count_1h": T.cart_count(evts, 3600, ref_time),
                "total_spend_24h": T.total_spend(evts, 86400, ref_time),
                "avg_session_duration": T.avg_session_duration(evts, ref_time),
                "conversion_rate_7d": T.conversion_rate(evts, 7 * 86400, ref_time),
                "category_affinity": T.category_affinity(evts, 86400, ref_time),
                "days_since_last_purchase": T.days_since_last_purchase(evts, ref_time),
            }
            records.append(
                FeatureRecord(entity="user", entity_id=user_id, features=features, timestamp=ref_time)
            )

        for item_id, evts in item_events.items():
            view_count_1h = item_view_counts.get(item_id, 0)
            popularity_rank = view_count_1h / max_views if max_views > 0 else 0.0
            features = {
                "view_count_1h": view_count_1h,
                "view_count_24h": T.item_view_count_for_item(evts, item_id, 86400, ref_time),
                "purchase_count_24h": T.item_purchase_count(evts, item_id, 86400, ref_time),
                "cart_add_count_1h": T.item_cart_count(evts, item_id, 3600, ref_time),
                "avg_rating": T.item_avg_rating(evts, item_id, ref_time),
                "conversion_rate_24h": T.item_conversion_rate(evts, item_id, 86400, ref_time),
                "revenue_24h": T.item_revenue(evts, item_id, 86400, ref_time),
                "popularity_rank_1h": round(popularity_rank, 6),
            }
            records.append(
                FeatureRecord(entity="item", entity_id=item_id, features=features, timestamp=ref_time)
            )

        if records:
            self._offline.write_batch(records)
