from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    entity: str          # "user" or "item"
    dtype: str           # "float", "int", "list"
    description: str
    ttl_seconds: int
    window: Optional[str]  # "1h", "24h", "7d", or None for all-time


USER_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        "purchase_count_1h", "user", "int",
        "Purchases in last 1 hour", 3600, "1h",
    ),
    FeatureDefinition(
        "purchase_count_24h", "user", "int",
        "Purchases in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "item_view_count_1h", "user", "int",
        "Item views in last 1 hour", 3600, "1h",
    ),
    FeatureDefinition(
        "item_view_count_24h", "user", "int",
        "Item views in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "cart_count_1h", "user", "int",
        "Add-to-cart events in last 1 hour", 3600, "1h",
    ),
    FeatureDefinition(
        "total_spend_24h", "user", "float",
        "Total purchase amount in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "avg_session_duration", "user", "float",
        "Average session duration in minutes", 86400, None,
    ),
    FeatureDefinition(
        "conversion_rate_7d", "user", "float",
        "Purchase / item_view ratio in last 7 days", 604800, "7d",
    ),
    FeatureDefinition(
        "category_affinity", "user", "list",
        "Top 3 item categories by view count in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "days_since_last_purchase", "user", "float",
        "Days since most recent purchase", 86400, None,
    ),
]

ITEM_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition(
        "view_count_1h", "item", "int",
        "Item views in last 1 hour", 3600, "1h",
    ),
    FeatureDefinition(
        "view_count_24h", "item", "int",
        "Item views in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "purchase_count_24h", "item", "int",
        "Purchases in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "cart_add_count_1h", "item", "int",
        "Add-to-cart events in last 1 hour", 3600, "1h",
    ),
    FeatureDefinition(
        "avg_rating", "item", "float",
        "Average user rating", 86400, None,
    ),
    FeatureDefinition(
        "conversion_rate_24h", "item", "float",
        "Purchase / view ratio in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "revenue_24h", "item", "float",
        "Total revenue in last 24 hours", 86400, "24h",
    ),
    FeatureDefinition(
        "popularity_rank_1h", "item", "float",
        "Relative popularity rank in last 1 hour (0=least, 1=most)", 3600, "1h",
    ),
]
