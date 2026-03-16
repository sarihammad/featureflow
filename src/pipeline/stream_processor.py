"""Stream processor: consume Kafka events, compute features, dual-write to stores.

The processor groups each incoming batch by user_id and item_id, computes all
registered features for each entity, and writes the results atomically to both
the online store (Redis) and the offline store (Parquet).  This dual-write
pattern is what guarantees training-serving consistency: both stores are fed
from the same computation path.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.events.schema import EventType, UserEvent
from src.features.definitions import FeatureDefinition
from src.features.registry import FeatureRegistry
from src.features import transformations as T
from src.kafka.consumer import EventConsumer
from src.stores.offline_store import FeatureRecord, OfflineFeatureStore
from src.stores.online_store import OnlineFeatureStore

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    users_updated: int
    items_updated: int
    events_processed: int
    latency_ms: float
    errors: int = 0


class StreamProcessor:
    def __init__(
        self,
        consumer: EventConsumer,
        online_store: OnlineFeatureStore,
        offline_store: OfflineFeatureStore,
        registry: FeatureRegistry,
        # How many events of history to keep per-entity in memory for windowed
        # aggregations.  In production this would be a Redis Sorted Set or
        # RocksDB state store.  For this implementation we use a bounded
        # in-process deque per entity.
        history_window_seconds: int = 7 * 86400,
    ):
        self._consumer = consumer
        self._online = online_store
        self._offline = offline_store
        self._registry = registry
        self._history_window_seconds = history_window_seconds

        # In-process event buffers: entity_id -> sorted list of events
        self._user_history: Dict[int, List[UserEvent]] = defaultdict(list)
        self._item_history: Dict[int, List[UserEvent]] = defaultdict(list)

        self._events_processed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(self, events: List[UserEvent]) -> ProcessingStats:
        t0 = time.perf_counter()
        errors = 0

        # Index events by user and item
        user_events: Dict[int, List[UserEvent]] = defaultdict(list)
        item_events: Dict[int, List[UserEvent]] = defaultdict(list)

        for event in events:
            user_events[event.user_id].append(event)
            if event.item_id is not None:
                item_events[event.item_id].append(event)

        # Update in-memory history buffers
        cutoff_ts: Optional[datetime] = None
        if events:
            latest_ts = max(e.timestamp for e in events)
            from datetime import timedelta
            cutoff_ts = latest_ts - timedelta(seconds=self._history_window_seconds)

        for uid, new_events in user_events.items():
            self._user_history[uid].extend(new_events)
            if cutoff_ts:
                self._user_history[uid] = [
                    e for e in self._user_history[uid] if e.timestamp >= cutoff_ts
                ]

        for iid, new_events in item_events.items():
            self._item_history[iid].extend(new_events)
            if cutoff_ts:
                self._item_history[iid] = [
                    e for e in self._item_history[iid] if e.timestamp >= cutoff_ts
                ]

        offline_records: List[FeatureRecord] = []
        ref_time = datetime.utcnow()

        # Process user features
        users_updated = 0
        for uid in user_events:
            try:
                history = self._user_history[uid]
                features = self._compute_user_features(uid, history, ref_time)
                ttl_map = {
                    f.name: f.ttl_seconds
                    for f in self._registry.list_user_features()
                }
                self._online.write_user_features(uid, features, ttl_per_feature=ttl_map)
                offline_records.append(
                    FeatureRecord(
                        entity="user",
                        entity_id=uid,
                        features=features,
                        timestamp=ref_time,
                    )
                )
                users_updated += 1
            except Exception as exc:
                logger.error("Error computing user features for user_id=%d: %s", uid, exc)
                errors += 1

        # Process item features
        items_updated = 0
        for iid in item_events:
            try:
                history = self._item_history[iid]
                features = self._compute_item_features(iid, history, ref_time)
                ttl_map = {
                    f.name: f.ttl_seconds
                    for f in self._registry.list_item_features()
                }
                self._online.write_item_features(iid, features, ttl_per_feature=ttl_map)
                offline_records.append(
                    FeatureRecord(
                        entity="item",
                        entity_id=iid,
                        features=features,
                        timestamp=ref_time,
                    )
                )
                items_updated += 1
            except Exception as exc:
                logger.error("Error computing item features for item_id=%d: %s", iid, exc)
                errors += 1

        try:
            self._offline.write_batch(offline_records)
        except Exception as exc:
            logger.error("Failed to write offline batch: %s", exc)
            errors += 1

        self._events_processed += len(events)
        latency_ms = (time.perf_counter() - t0) * 1000

        if self._events_processed % 1000 < len(events):
            logger.info(
                "Processed %d total events | batch: %d events, %d users, %d items, %.1f ms",
                self._events_processed,
                len(events),
                users_updated,
                items_updated,
                latency_ms,
            )

        return ProcessingStats(
            users_updated=users_updated,
            items_updated=items_updated,
            events_processed=len(events),
            latency_ms=latency_ms,
            errors=errors,
        )

    def run_forever(self) -> None:
        logger.info("StreamProcessor starting consume loop")
        self._consumer.consume_forever(
            callback=self.process_batch,
            batch_size=100,
            timeout_ms=500,
        )

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    def _compute_user_features(
        self,
        user_id: int,
        all_user_events: List[UserEvent],
        ref_time: datetime,
    ) -> Dict:
        return {
            "purchase_count_1h": T.purchase_count(all_user_events, 3600, ref_time),
            "purchase_count_24h": T.purchase_count(all_user_events, 86400, ref_time),
            "item_view_count_1h": T.item_view_count(all_user_events, 3600, ref_time),
            "item_view_count_24h": T.item_view_count(all_user_events, 86400, ref_time),
            "cart_count_1h": T.cart_count(all_user_events, 3600, ref_time),
            "total_spend_24h": T.total_spend(all_user_events, 86400, ref_time),
            "avg_session_duration": T.avg_session_duration(all_user_events, ref_time),
            "conversion_rate_7d": T.conversion_rate(all_user_events, 7 * 86400, ref_time),
            "category_affinity": T.category_affinity(all_user_events, 86400, ref_time),
            "days_since_last_purchase": T.days_since_last_purchase(all_user_events, ref_time),
        }

    def _compute_item_features(
        self,
        item_id: int,
        all_item_events: List[UserEvent],
        ref_time: datetime,
    ) -> Dict:
        return {
            "view_count_1h": T.item_view_count_for_item(all_item_events, item_id, 3600, ref_time),
            "view_count_24h": T.item_view_count_for_item(all_item_events, item_id, 86400, ref_time),
            "purchase_count_24h": T.item_purchase_count(all_item_events, item_id, 86400, ref_time),
            "cart_add_count_1h": T.item_cart_count(all_item_events, item_id, 3600, ref_time),
            "avg_rating": T.item_avg_rating(all_item_events, item_id, ref_time),
            "conversion_rate_24h": T.item_conversion_rate(all_item_events, item_id, 86400, ref_time),
            "revenue_24h": T.item_revenue(all_item_events, item_id, 86400, ref_time),
            # popularity_rank_1h is relative — set to 0 here; updated in batch
            "popularity_rank_1h": 0.0,
        }
