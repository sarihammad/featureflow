import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional

import numpy as np
from faker import Faker

from src.events.schema import EventType, UserEvent

fake = Faker()

ITEM_CATEGORIES = [
    "electronics",
    "clothing",
    "books",
    "home_garden",
    "sports",
    "beauty",
    "toys",
    "food",
    "automotive",
    "jewelry",
]

EVENT_TYPE_WEIGHTS = {
    EventType.PAGE_VIEW: 0.40,
    EventType.ITEM_VIEW: 0.30,
    EventType.ADD_TO_CART: 0.15,
    EventType.PURCHASE: 0.10,
    EventType.SEARCH: 0.04,
    EventType.RATING: 0.01,
}


class UserProfile:
    __slots__ = ("user_id", "category_affinities", "session_id", "session_start", "purchase_history")

    def __init__(self, user_id: int, n_categories: int = 10):
        self.user_id = user_id
        raw = np.random.dirichlet(np.ones(n_categories) * 0.5)
        self.category_affinities: np.ndarray = raw
        self.session_id: str = str(uuid.uuid4())
        self.session_start: datetime = datetime.utcnow()
        self.purchase_history: List[datetime] = []

    def refresh_session(self) -> None:
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.utcnow()


class ItemCatalog:
    __slots__ = ("item_id", "category", "price", "popularity")

    def __init__(self, item_id: int):
        self.item_id = item_id
        self.category = random.choice(ITEM_CATEGORIES)
        self.price = round(random.uniform(10.0, 500.0), 2)
        self.popularity = random.betavariate(2, 5)  # right-skewed: most items are less popular


class EventGenerator:
    def __init__(
        self,
        n_users: int = 10_000,
        n_items: int = 5_000,
        events_per_second: float = 100.0,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_users = n_users
        self.n_items = n_items
        self.events_per_second = events_per_second

        self._users: Dict[int, UserProfile] = {
            uid: UserProfile(uid) for uid in range(1, n_users + 1)
        }
        self._items: Dict[int, ItemCatalog] = {
            iid: ItemCatalog(iid) for iid in range(1, n_items + 1)
        }

        # Precompute per-category item lists for fast sampling
        self._items_by_category: Dict[str, List[int]] = {cat: [] for cat in ITEM_CATEGORIES}
        for item in self._items.values():
            self._items_by_category[item.category].append(item.item_id)

        self._event_types = list(EVENT_TYPE_WEIGHTS.keys())
        self._event_weights = list(EVENT_TYPE_WEIGHTS.values())

        # Session expiry after ~20 minutes of inactivity (simulated)
        self._session_ttl_seconds = 1200

    def _sample_item_for_user(self, user: UserProfile) -> int:
        chosen_category = ITEM_CATEGORIES[
            np.random.choice(len(ITEM_CATEGORIES), p=user.category_affinities)
        ]
        candidates = self._items_by_category[chosen_category]
        if not candidates:
            return random.randint(1, self.n_items)
        popularities = np.array([self._items[i].popularity for i in candidates])
        popularities /= popularities.sum()
        return int(np.random.choice(candidates, p=popularities))

    def _maybe_new_session(self, user: UserProfile, now: datetime) -> None:
        elapsed = (now - user.session_start).total_seconds()
        if elapsed > self._session_ttl_seconds or random.random() < 0.001:
            user.refresh_session()

    def generate_event(
        self,
        user_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> UserEvent:
        if user_id is None:
            user_id = random.randint(1, self.n_users)

        user = self._users[user_id]
        now = timestamp or datetime.utcnow()
        self._maybe_new_session(user, now)

        event_type: EventType = random.choices(self._event_types, weights=self._event_weights, k=1)[0]

        item_id: Optional[int] = None
        metadata: Dict = {}

        if event_type in (EventType.ITEM_VIEW, EventType.ADD_TO_CART, EventType.PURCHASE, EventType.RATING):
            item_id = self._sample_item_for_user(user)
            item = self._items[item_id]
            metadata["price"] = item.price
            metadata["category"] = item.category

            if event_type == EventType.PURCHASE:
                quantity = random.randint(1, 3)
                metadata["quantity"] = quantity
                metadata["total_amount"] = round(item.price * quantity, 2)
                user.purchase_history.append(now)

            elif event_type == EventType.RATING:
                metadata["rating"] = round(random.gauss(3.8, 0.9), 1)
                metadata["rating"] = max(1.0, min(5.0, metadata["rating"]))

        elif event_type == EventType.SEARCH:
            metadata["query"] = fake.word() + " " + random.choice(ITEM_CATEGORIES)
            metadata["results_count"] = random.randint(0, 200)

        elif event_type == EventType.PAGE_VIEW:
            metadata["page"] = random.choice(["home", "category", "cart", "checkout", "wishlist", "search"])

        return UserEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            event_type=event_type,
            item_id=item_id,
            timestamp=now,
            session_id=user.session_id,
            metadata=metadata,
        )

    def generate_batch(self, n: int, start_time: Optional[datetime] = None) -> List[UserEvent]:
        events = []
        base_time = start_time or datetime.utcnow()
        for i in range(n):
            jitter = timedelta(seconds=i / max(self.events_per_second, 1.0))
            event = self.generate_event(timestamp=base_time + jitter)
            events.append(event)
        return events

    def generate_historical(
        self,
        n_events: int,
        end_time: Optional[datetime] = None,
        span_hours: float = 48.0,
    ) -> List[UserEvent]:
        end = end_time or datetime.utcnow()
        start = end - timedelta(hours=span_hours)
        events = []
        for _ in range(n_events):
            ts = start + timedelta(seconds=random.uniform(0, span_hours * 3600))
            event = self.generate_event(timestamp=ts)
            events.append(event)
        events.sort(key=lambda e: e.timestamp)
        return events

    def stream(
        self,
        duration_seconds: Optional[float] = None,
    ) -> Iterator[UserEvent]:
        interval = 1.0 / self.events_per_second
        start = time.monotonic()
        while True:
            if duration_seconds is not None:
                elapsed = time.monotonic() - start
                if elapsed >= duration_seconds:
                    break
            yield self.generate_event()
            time.sleep(interval)
