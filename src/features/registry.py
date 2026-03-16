"""Feature registry — a singleton catalog of all defined FeatureDefinition objects.

Importing this module auto-registers all USER_FEATURES and ITEM_FEATURES so
that downstream components (stream processor, serving API, consistency checker)
can look up feature metadata by name without hard-coded lists.
"""

from typing import Dict, List

from src.features.definitions import (
    ITEM_FEATURES,
    USER_FEATURES,
    FeatureDefinition,
)


class FeatureRegistry:
    """Thread-safe singleton registry for feature metadata."""

    _instance: "FeatureRegistry | None" = None

    def __new__(cls) -> "FeatureRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._features: Dict[str, FeatureDefinition] = {}
        return cls._instance

    def register(self, feature: FeatureDefinition) -> None:
        if feature.name in self._features:
            raise ValueError(f"Feature '{feature.name}' is already registered.")
        self._features[feature.name] = feature

    def get(self, name: str) -> FeatureDefinition:
        try:
            return self._features[name]
        except KeyError:
            raise KeyError(f"Feature '{name}' is not registered.") from None

    def list_user_features(self) -> List[FeatureDefinition]:
        return [f for f in self._features.values() if f.entity == "user"]

    def list_item_features(self) -> List[FeatureDefinition]:
        return [f for f in self._features.values() if f.entity == "item"]

    def list_all(self) -> List[FeatureDefinition]:
        return list(self._features.values())

    def __contains__(self, name: str) -> bool:
        return name in self._features


# ---------------------------------------------------------------------------
# Auto-register all built-in features on module import.
# ---------------------------------------------------------------------------

_registry = FeatureRegistry()

for _feature in USER_FEATURES + ITEM_FEATURES:
    if _feature.name not in _registry:
        _registry.register(_feature)
