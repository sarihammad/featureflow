"""Redis-backed online feature store.

Key schema:   features:{entity}:{entity_id}:{feature_name}
Value schema: JSON-encoded scalar or list

When Redis is unavailable the store falls back to an in-process dict so
that unit tests and local development runs that lack a Redis instance can
still exercise the full pipeline.  A warning is logged when the fallback
is active.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import redis as _redis

logger = logging.getLogger(__name__)

_FALLBACK_STORE: Dict[str, str] = {}


def _key(entity: str, entity_id: int, feature_name: str) -> str:
    return f"features:{entity}:{entity_id}:{feature_name}"


class OnlineFeatureStore:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 86400,
    ):
        self._default_ttl = default_ttl
        self._redis: Optional[_redis.Redis] = None
        self._fallback_active = False

        try:
            client = _redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            logger.info("OnlineFeatureStore connected to Redis at %s", redis_url)
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s) — using in-memory fallback. "
                "TTL and persistence are NOT honoured in fallback mode.",
                exc,
            )
            self._fallback_active = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pipeline_set(
        self,
        kv_pairs: Dict[str, str],
        ttl_map: Dict[str, int],
    ) -> None:
        if self._fallback_active:
            _FALLBACK_STORE.update(kv_pairs)
            return
        pipe = self._redis.pipeline(transaction=False)
        for k, v in kv_pairs.items():
            ttl = ttl_map.get(k, self._default_ttl)
            pipe.set(k, v, ex=ttl)
        pipe.execute()

    def _pipeline_get(self, keys: List[str]) -> List[Optional[str]]:
        if self._fallback_active:
            return [_FALLBACK_STORE.get(k) for k in keys]
        pipe = self._redis.pipeline(transaction=False)
        for k in keys:
            pipe.get(k)
        return pipe.execute()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_user_features(
        self,
        user_id: int,
        features: Dict[str, Any],
        ttl_per_feature: Optional[Dict[str, int]] = None,
    ) -> None:
        kv: Dict[str, str] = {}
        ttl_map: Dict[str, int] = {}
        for fname, value in features.items():
            k = _key("user", user_id, fname)
            kv[k] = json.dumps(value)
            ttl_map[k] = (ttl_per_feature or {}).get(fname, self._default_ttl)
        self._pipeline_set(kv, ttl_map)

    def write_item_features(
        self,
        item_id: int,
        features: Dict[str, Any],
        ttl_per_feature: Optional[Dict[str, int]] = None,
    ) -> None:
        kv: Dict[str, str] = {}
        ttl_map: Dict[str, int] = {}
        for fname, value in features.items():
            k = _key("item", item_id, fname)
            kv[k] = json.dumps(value)
            ttl_map[k] = (ttl_per_feature or {}).get(fname, self._default_ttl)
        self._pipeline_set(kv, ttl_map)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def read_user_features(
        self,
        user_id: int,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        keys = [_key("user", user_id, f) for f in feature_names]
        raw_values = self._pipeline_get(keys)
        result: Dict[str, Any] = {}
        for fname, raw in zip(feature_names, raw_values):
            result[fname] = json.loads(raw) if raw is not None else None
        return result

    def read_item_features(
        self,
        item_id: int,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        keys = [_key("item", item_id, f) for f in feature_names]
        raw_values = self._pipeline_get(keys)
        result: Dict[str, Any] = {}
        for fname, raw in zip(feature_names, raw_values):
            result[fname] = json.loads(raw) if raw is not None else None
        return result

    def read_feature_vector(
        self,
        user_id: int,
        item_id: int,
        user_feature_names: List[str],
        item_feature_names: List[str],
    ) -> Dict[str, Any]:
        """Fetch user and item features in a single pipeline call.

        Returns a merged dict with ``user_`` and ``item_`` prefixes to
        avoid name collisions between entities.
        """
        user_keys = [_key("user", user_id, f) for f in user_feature_names]
        item_keys = [_key("item", item_id, f) for f in item_feature_names]
        all_keys = user_keys + item_keys
        raw_values = self._pipeline_get(all_keys)

        result: Dict[str, Any] = {}
        for fname, raw in zip(user_feature_names, raw_values[: len(user_feature_names)]):
            result[f"user_{fname}"] = json.loads(raw) if raw is not None else None
        for fname, raw in zip(item_feature_names, raw_values[len(user_feature_names) :]):
            result[f"item_{fname}"] = json.loads(raw) if raw is not None else None
        return result

    # ------------------------------------------------------------------
    # Delete / admin
    # ------------------------------------------------------------------

    def delete_user_features(self, user_id: int) -> None:
        if self._fallback_active:
            prefix = f"features:user:{user_id}:"
            keys = [k for k in list(_FALLBACK_STORE) if k.startswith(prefix)]
            for k in keys:
                del _FALLBACK_STORE[k]
            return
        pattern = f"features:user:{user_id}:*"
        keys = self._redis.keys(pattern)
        if keys:
            self._redis.delete(*keys)

    def delete_item_features(self, item_id: int) -> None:
        if self._fallback_active:
            prefix = f"features:item:{item_id}:"
            keys = [k for k in list(_FALLBACK_STORE) if k.startswith(prefix)]
            for k in keys:
                del _FALLBACK_STORE[k]
            return
        pattern = f"features:item:{item_id}:*"
        keys = self._redis.keys(pattern)
        if keys:
            self._redis.delete(*keys)

    def get_ttl(self, entity: str, entity_id: int, feature_name: str) -> Optional[int]:
        """Return remaining TTL in seconds for a key, or None if not present."""
        if self._fallback_active:
            return None
        k = _key(entity, entity_id, feature_name)
        ttl = self._redis.ttl(k)
        return ttl if ttl >= 0 else None

    def health_check(self) -> bool:
        if self._fallback_active:
            return True  # fallback is always "healthy"
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
