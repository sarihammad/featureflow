"""Unit tests for feature transformation functions.

All tests are pure: no I/O, no Redis, no Kafka.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.events.schema import EventType, UserEvent
from src.features import transformations as T


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
ONE_HOUR = 3600
ONE_DAY = 86400
SEVEN_DAYS = 7 * 86400


def make_event(
    event_type: EventType,
    seconds_before_ref: float,
    user_id: int = 1,
    item_id: int = 101,
    session_id: str = "sess-1",
    metadata: dict = None,
) -> UserEvent:
    return UserEvent(
        event_id=f"e-{event_type.value}-{seconds_before_ref}",
        user_id=user_id,
        event_type=event_type,
        item_id=item_id if event_type != EventType.PAGE_VIEW else None,
        timestamp=BASE_TIME - timedelta(seconds=seconds_before_ref),
        session_id=session_id,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# purchase_count
# ---------------------------------------------------------------------------


class TestPurchaseCount:
    def test_counts_events_within_window(self):
        events = [
            make_event(EventType.PURCHASE, 100),   # 100s ago — inside 1h
            make_event(EventType.PURCHASE, 3500),  # 3500s ago — inside 1h
            make_event(EventType.PURCHASE, 3700),  # 3700s ago — outside 1h
        ]
        assert T.purchase_count(events, ONE_HOUR, BASE_TIME) == 2

    def test_excludes_events_exactly_at_window_boundary(self):
        # Window is [ref - window, ref) — event at exactly ref - window is INCLUDED
        events = [make_event(EventType.PURCHASE, ONE_HOUR)]
        assert T.purchase_count(events, ONE_HOUR, BASE_TIME) == 1

    def test_excludes_events_after_ref_time(self):
        future_event = UserEvent(
            event_id="future",
            user_id=1,
            event_type=EventType.PURCHASE,
            item_id=None,
            timestamp=BASE_TIME + timedelta(seconds=10),
            session_id="s",
            metadata={},
        )
        assert T.purchase_count([future_event], ONE_HOUR, BASE_TIME) == 0

    def test_ignores_non_purchase_events(self):
        events = [
            make_event(EventType.ITEM_VIEW, 100),
            make_event(EventType.ADD_TO_CART, 200),
            make_event(EventType.PURCHASE, 300),
        ]
        assert T.purchase_count(events, ONE_HOUR, BASE_TIME) == 1

    def test_empty_events_returns_zero(self):
        assert T.purchase_count([], ONE_HOUR, BASE_TIME) == 0

    def test_24h_window_counts_more_than_1h(self):
        events = [
            make_event(EventType.PURCHASE, 1800),    # 0.5h ago — in 1h and 24h
            make_event(EventType.PURCHASE, 7200),    # 2h ago — in 24h but NOT 1h
            make_event(EventType.PURCHASE, 90000),   # 25h ago — outside 24h
        ]
        assert T.purchase_count(events, ONE_HOUR, BASE_TIME) == 1
        assert T.purchase_count(events, ONE_DAY, BASE_TIME) == 2


# ---------------------------------------------------------------------------
# item_view_count
# ---------------------------------------------------------------------------


class TestItemViewCount:
    def test_counts_item_views_in_window(self):
        events = [
            make_event(EventType.ITEM_VIEW, 500),
            make_event(EventType.ITEM_VIEW, 1000),
            make_event(EventType.ITEM_VIEW, 5000),  # outside 1h
        ]
        assert T.item_view_count(events, ONE_HOUR, BASE_TIME) == 2

    def test_zero_when_no_item_views(self):
        events = [make_event(EventType.PAGE_VIEW, 100)]
        assert T.item_view_count(events, ONE_HOUR, BASE_TIME) == 0


# ---------------------------------------------------------------------------
# cart_count
# ---------------------------------------------------------------------------


class TestCartCount:
    def test_counts_add_to_cart_in_window(self):
        events = [
            make_event(EventType.ADD_TO_CART, 60),
            make_event(EventType.ADD_TO_CART, 120),
            make_event(EventType.ITEM_VIEW, 60),
        ]
        assert T.cart_count(events, ONE_HOUR, BASE_TIME) == 2


# ---------------------------------------------------------------------------
# total_spend
# ---------------------------------------------------------------------------


class TestTotalSpend:
    def test_sums_total_amount_field(self):
        events = [
            make_event(EventType.PURCHASE, 100, metadata={"total_amount": 49.99, "price": 49.99}),
            make_event(EventType.PURCHASE, 200, metadata={"total_amount": 150.00, "price": 150.00}),
        ]
        result = T.total_spend(events, ONE_DAY, BASE_TIME)
        assert abs(result - 199.99) < 0.001

    def test_falls_back_to_price_when_no_total_amount(self):
        events = [
            make_event(EventType.PURCHASE, 100, metadata={"price": 25.0}),
        ]
        assert T.total_spend(events, ONE_DAY, BASE_TIME) == pytest.approx(25.0)

    def test_zero_when_no_purchases(self):
        events = [make_event(EventType.ITEM_VIEW, 100, metadata={})]
        assert T.total_spend(events, ONE_DAY, BASE_TIME) == 0.0


# ---------------------------------------------------------------------------
# conversion_rate
# ---------------------------------------------------------------------------


class TestConversionRate:
    def test_correct_ratio(self):
        events = [
            make_event(EventType.ITEM_VIEW, 100),
            make_event(EventType.ITEM_VIEW, 200),
            make_event(EventType.ITEM_VIEW, 300),
            make_event(EventType.ITEM_VIEW, 400),
            make_event(EventType.PURCHASE, 500, metadata={"total_amount": 10.0}),
        ]
        rate = T.conversion_rate(events, ONE_HOUR, BASE_TIME)
        assert rate == pytest.approx(0.25, abs=1e-6)

    def test_no_division_by_zero_when_no_views(self):
        events = [make_event(EventType.PURCHASE, 100, metadata={"total_amount": 10.0})]
        assert T.conversion_rate(events, ONE_HOUR, BASE_TIME) == 0.0

    def test_zero_when_no_events(self):
        assert T.conversion_rate([], ONE_HOUR, BASE_TIME) == 0.0


# ---------------------------------------------------------------------------
# avg_session_duration
# ---------------------------------------------------------------------------


class TestAvgSessionDuration:
    def test_single_session_duration(self):
        events = [
            UserEvent("e1", 1, EventType.PAGE_VIEW, None, BASE_TIME - timedelta(minutes=30), "sess-A", {}),
            UserEvent("e2", 1, EventType.ITEM_VIEW, 1, BASE_TIME - timedelta(minutes=20), "sess-A", {}),
            UserEvent("e3", 1, EventType.PURCHASE, 1, BASE_TIME - timedelta(minutes=10), "sess-A", {"total_amount": 1.0}),
        ]
        # Session A spans 20 minutes
        result = T.avg_session_duration(events, BASE_TIME)
        assert abs(result - 20.0) < 0.01

    def test_multiple_sessions_averaged(self):
        events = [
            UserEvent("e1", 1, EventType.PAGE_VIEW, None, BASE_TIME - timedelta(minutes=60), "sess-A", {}),
            UserEvent("e2", 1, EventType.PAGE_VIEW, None, BASE_TIME - timedelta(minutes=40), "sess-A", {}),
            # sess-A: 20 min
            UserEvent("e3", 1, EventType.PAGE_VIEW, None, BASE_TIME - timedelta(minutes=30), "sess-B", {}),
            UserEvent("e4", 1, EventType.PAGE_VIEW, None, BASE_TIME - timedelta(minutes=20), "sess-B", {}),
            # sess-B: 10 min
        ]
        result = T.avg_session_duration(events, BASE_TIME)
        assert abs(result - 15.0) < 0.01

    def test_single_event_session_not_counted(self):
        events = [
            UserEvent("e1", 1, EventType.PAGE_VIEW, None, BASE_TIME - timedelta(minutes=10), "sess-A", {}),
        ]
        assert T.avg_session_duration(events, BASE_TIME) == 0.0


# ---------------------------------------------------------------------------
# days_since_last_purchase
# ---------------------------------------------------------------------------


class TestDaysSinceLastPurchase:
    def test_returns_correct_days(self):
        events = [
            make_event(EventType.PURCHASE, 2 * 86400, metadata={"total_amount": 10.0}),   # 2 days ago
            make_event(EventType.PURCHASE, 5 * 86400, metadata={"total_amount": 10.0}),   # 5 days ago
        ]
        result = T.days_since_last_purchase(events, BASE_TIME)
        assert abs(result - 2.0) < 0.001

    def test_returns_inf_when_no_purchases(self):
        events = [make_event(EventType.ITEM_VIEW, 100)]
        result = T.days_since_last_purchase(events, BASE_TIME)
        assert math.isinf(result)

    def test_empty_events_returns_inf(self):
        assert math.isinf(T.days_since_last_purchase([], BASE_TIME))


# ---------------------------------------------------------------------------
# category_affinity
# ---------------------------------------------------------------------------


class TestCategoryAffinity:
    def test_returns_top_n_categories(self):
        events = [
            make_event(EventType.ITEM_VIEW, 100, metadata={"category": "electronics"}),
            make_event(EventType.ITEM_VIEW, 200, metadata={"category": "electronics"}),
            make_event(EventType.ITEM_VIEW, 300, metadata={"category": "electronics"}),
            make_event(EventType.ITEM_VIEW, 400, metadata={"category": "clothing"}),
            make_event(EventType.ITEM_VIEW, 500, metadata={"category": "clothing"}),
            make_event(EventType.ITEM_VIEW, 600, metadata={"category": "books"}),
        ]
        result = T.category_affinity(events, ONE_DAY, BASE_TIME, top_n=2)
        assert result[0] == "electronics"
        assert result[1] == "clothing"
        assert len(result) == 2

    def test_returns_empty_when_no_views(self):
        events = [make_event(EventType.PURCHASE, 100, metadata={"total_amount": 10.0})]
        result = T.category_affinity(events, ONE_DAY, BASE_TIME)
        assert result == []

    def test_respects_window(self):
        events = [
            make_event(EventType.ITEM_VIEW, 100, metadata={"category": "electronics"}),
            make_event(EventType.ITEM_VIEW, ONE_DAY + 100, metadata={"category": "clothing"}),
        ]
        # Only events within 1h window
        result = T.category_affinity(events, ONE_HOUR, BASE_TIME, top_n=3)
        assert result == ["electronics"]

    def test_top_n_respected(self):
        events = [
            make_event(EventType.ITEM_VIEW, 100 + i * 10, metadata={"category": cat})
            for i, cat in enumerate(["a", "b", "c", "d", "e"])
        ]
        result = T.category_affinity(events, ONE_DAY, BASE_TIME, top_n=3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# Item feature functions
# ---------------------------------------------------------------------------


class TestItemFeatures:
    def _make_item_events(self):
        item_id = 42
        other_id = 99
        return item_id, [
            UserEvent("e1", 1, EventType.ITEM_VIEW, item_id, BASE_TIME - timedelta(seconds=100), "s", {}),
            UserEvent("e2", 2, EventType.ITEM_VIEW, item_id, BASE_TIME - timedelta(seconds=200), "s", {}),
            UserEvent("e3", 3, EventType.ITEM_VIEW, other_id, BASE_TIME - timedelta(seconds=300), "s", {}),
            UserEvent("e4", 1, EventType.PURCHASE, item_id, BASE_TIME - timedelta(seconds=400), "s", {"total_amount": 99.0}),
            UserEvent("e5", 2, EventType.ADD_TO_CART, item_id, BASE_TIME - timedelta(seconds=500), "s", {"price": 50.0}),
            UserEvent("e6", 3, EventType.RATING, item_id, BASE_TIME - timedelta(seconds=600), "s", {"rating": 4.5}),
        ]

    def test_item_view_count_filters_by_item(self):
        item_id, events = self._make_item_events()
        assert T.item_view_count_for_item(events, item_id, ONE_HOUR, BASE_TIME) == 2
        # other item should have 1 view
        assert T.item_view_count_for_item(events, 99, ONE_HOUR, BASE_TIME) == 1

    def test_item_purchase_count(self):
        item_id, events = self._make_item_events()
        assert T.item_purchase_count(events, item_id, ONE_HOUR, BASE_TIME) == 1

    def test_item_cart_count(self):
        item_id, events = self._make_item_events()
        assert T.item_cart_count(events, item_id, ONE_HOUR, BASE_TIME) == 1

    def test_item_avg_rating(self):
        item_id, events = self._make_item_events()
        assert T.item_avg_rating(events, item_id, BASE_TIME) == pytest.approx(4.5)

    def test_item_avg_rating_no_ratings(self):
        assert T.item_avg_rating([], 1, BASE_TIME) == 0.0

    def test_item_revenue(self):
        item_id, events = self._make_item_events()
        assert T.item_revenue(events, item_id, ONE_HOUR, BASE_TIME) == pytest.approx(99.0)

    def test_item_conversion_rate_no_division_by_zero(self):
        # Item with 0 views
        assert T.item_conversion_rate([], 42, ONE_HOUR, BASE_TIME) == 0.0

    def test_item_conversion_rate_correct(self):
        item_id = 42
        events = [
            UserEvent("e1", 1, EventType.ITEM_VIEW, item_id, BASE_TIME - timedelta(seconds=100), "s", {}),
            UserEvent("e2", 1, EventType.ITEM_VIEW, item_id, BASE_TIME - timedelta(seconds=200), "s", {}),
            UserEvent("e3", 1, EventType.PURCHASE, item_id, BASE_TIME - timedelta(seconds=300), "s", {"total_amount": 10.0}),
        ]
        # 1 purchase / 2 views = 0.5
        assert T.item_conversion_rate(events, item_id, ONE_HOUR, BASE_TIME) == pytest.approx(0.5)
