"""Pure, stateless feature computation functions.

Every function here accepts a list of UserEvent objects and a reference
timestamp, filters to the relevant time window, and returns a scalar or
list value.  None of these functions mutate state or perform I/O — they
are safe to call from any thread and can be trivially unit-tested.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.events.schema import EventType, UserEvent


def _window_start(ref_time: datetime, window_seconds: int) -> datetime:
    return ref_time - timedelta(seconds=window_seconds)


def _in_window(event: UserEvent, window_seconds: int, ref_time: datetime) -> bool:
    start = _window_start(ref_time, window_seconds)
    return start <= event.timestamp < ref_time


# ---------------------------------------------------------------------------
# User features
# ---------------------------------------------------------------------------


def purchase_count(
    events: List[UserEvent],
    window_seconds: int,
    ref_time: datetime,
) -> int:
    return sum(
        1
        for e in events
        if e.event_type == EventType.PURCHASE and _in_window(e, window_seconds, ref_time)
    )


def item_view_count(
    events: List[UserEvent],
    window_seconds: int,
    ref_time: datetime,
) -> int:
    return sum(
        1
        for e in events
        if e.event_type == EventType.ITEM_VIEW and _in_window(e, window_seconds, ref_time)
    )


def cart_count(
    events: List[UserEvent],
    window_seconds: int,
    ref_time: datetime,
) -> int:
    return sum(
        1
        for e in events
        if e.event_type == EventType.ADD_TO_CART and _in_window(e, window_seconds, ref_time)
    )


def total_spend(
    events: List[UserEvent],
    window_seconds: int,
    ref_time: datetime,
) -> float:
    total = 0.0
    for e in events:
        if e.event_type == EventType.PURCHASE and _in_window(e, window_seconds, ref_time):
            total += float(e.metadata.get("total_amount", e.metadata.get("price", 0.0)))
    return round(total, 4)


def conversion_rate(
    events: List[UserEvent],
    window_seconds: int,
    ref_time: datetime,
) -> float:
    views = item_view_count(events, window_seconds, ref_time)
    purchases = purchase_count(events, window_seconds, ref_time)
    if views == 0:
        return 0.0
    return round(purchases / views, 6)


def avg_session_duration(
    events: List[UserEvent],
    ref_time: datetime,
) -> float:
    """Return mean session duration in minutes across all sessions in the event list."""
    session_times: Dict[str, List[datetime]] = defaultdict(list)
    for e in events:
        if e.timestamp < ref_time:
            session_times[e.session_id].append(e.timestamp)

    durations: List[float] = []
    for timestamps in session_times.values():
        if len(timestamps) < 2:
            continue
        duration_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60.0
        durations.append(duration_minutes)

    if not durations:
        return 0.0
    return round(sum(durations) / len(durations), 4)


def days_since_last_purchase(
    events: List[UserEvent],
    ref_time: datetime,
) -> float:
    purchase_times = [
        e.timestamp
        for e in events
        if e.event_type == EventType.PURCHASE and e.timestamp < ref_time
    ]
    if not purchase_times:
        return float("inf")
    latest = max(purchase_times)
    return round((ref_time - latest).total_seconds() / 86400.0, 4)


def category_affinity(
    events: List[UserEvent],
    window_seconds: int,
    ref_time: datetime,
    top_n: int = 3,
) -> List[str]:
    counts: Counter = Counter()
    for e in events:
        if e.event_type == EventType.ITEM_VIEW and _in_window(e, window_seconds, ref_time):
            category = e.metadata.get("category")
            if category:
                counts[category] += 1
    return [cat for cat, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Item features
# ---------------------------------------------------------------------------


def item_view_count_for_item(
    events: List[UserEvent],
    item_id: int,
    window_seconds: int,
    ref_time: datetime,
) -> int:
    return sum(
        1
        for e in events
        if (
            e.event_type == EventType.ITEM_VIEW
            and e.item_id == item_id
            and _in_window(e, window_seconds, ref_time)
        )
    )


def item_purchase_count(
    events: List[UserEvent],
    item_id: int,
    window_seconds: int,
    ref_time: datetime,
) -> int:
    return sum(
        1
        for e in events
        if (
            e.event_type == EventType.PURCHASE
            and e.item_id == item_id
            and _in_window(e, window_seconds, ref_time)
        )
    )


def item_cart_count(
    events: List[UserEvent],
    item_id: int,
    window_seconds: int,
    ref_time: datetime,
) -> int:
    return sum(
        1
        for e in events
        if (
            e.event_type == EventType.ADD_TO_CART
            and e.item_id == item_id
            and _in_window(e, window_seconds, ref_time)
        )
    )


def item_avg_rating(
    events: List[UserEvent],
    item_id: int,
    ref_time: datetime,
) -> float:
    ratings = [
        float(e.metadata["rating"])
        for e in events
        if (
            e.event_type == EventType.RATING
            and e.item_id == item_id
            and e.timestamp < ref_time
            and "rating" in e.metadata
        )
    ]
    if not ratings:
        return 0.0
    return round(sum(ratings) / len(ratings), 4)


def item_revenue(
    events: List[UserEvent],
    item_id: int,
    window_seconds: int,
    ref_time: datetime,
) -> float:
    total = 0.0
    for e in events:
        if (
            e.event_type == EventType.PURCHASE
            and e.item_id == item_id
            and _in_window(e, window_seconds, ref_time)
        ):
            total += float(e.metadata.get("total_amount", e.metadata.get("price", 0.0)))
    return round(total, 4)


def item_conversion_rate(
    events: List[UserEvent],
    item_id: int,
    window_seconds: int,
    ref_time: datetime,
) -> float:
    views = item_view_count_for_item(events, item_id, window_seconds, ref_time)
    purchases = item_purchase_count(events, item_id, window_seconds, ref_time)
    if views == 0:
        return 0.0
    return round(purchases / views, 6)
