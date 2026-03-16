"""FastAPI feature serving API.

Hot path latency target: <5 ms for /features/vector (single Redis pipeline).

Endpoints
---------
GET  /features/user/{user_id}        – per-user feature lookup
GET  /features/item/{item_id}        – per-item feature lookup
GET  /features/vector                – merged user+item feature vector
POST /features/batch                 – multi-entity batch lookup
GET  /health                         – liveness probe
GET  /metrics                        – Prometheus metrics
GET  /registry                       – feature catalog
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.config import settings
from src.features.registry import FeatureRegistry, _registry
from src.stores.online_store import OnlineFeatureStore
from src.serving.middleware import (
    MetricsMiddleware,
    cache_miss_total,
    features_served_total,
    redis_lookup_latency,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (initialised in lifespan)
# ---------------------------------------------------------------------------

_online_store: Optional[OnlineFeatureStore] = None
_feature_registry: Optional[FeatureRegistry] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _online_store, _feature_registry
    _online_store = OnlineFeatureStore(
        redis_url=settings.REDIS_URL,
        default_ttl=settings.FEATURE_TTL_SECONDS,
    )
    _feature_registry = _registry
    logger.info("FeatureFlow API started. Redis health: %s", _online_store.health_check())
    yield
    logger.info("FeatureFlow API shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FeatureFlow",
    description="Real-time ML feature store serving API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(MetricsMiddleware)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class FeatureResponse(BaseModel):
    entity_type: str
    entity_id: int
    features: Dict[str, Any]
    missing_features: List[str]
    latency_ms: float


class VectorResponse(BaseModel):
    user_id: int
    item_id: int
    features: Dict[str, Any]
    missing_features: List[str]
    latency_ms: float


class BatchRequest(BaseModel):
    class EntityRequest(BaseModel):
        entity: str
        entity_id: int
        feature_names: List[str]

    requests: List[EntityRequest]


class BatchResponse(BaseModel):
    results: List[FeatureResponse]
    total_latency_ms: float


class RegistryEntry(BaseModel):
    name: str
    entity: str
    dtype: str
    description: str
    ttl_seconds: int
    window: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_feature_names(entity: str) -> List[str]:
    assert _feature_registry is not None
    return [f.name for f in _feature_registry.list_all() if f.entity == entity]


def _record_cache_misses(features: Dict[str, Any], entity_type: str) -> List[str]:
    missing = [k for k, v in features.items() if v is None]
    for fname in missing:
        cache_miss_total.labels(entity_type=entity_type, feature_name=fname).inc()
    return missing


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/features/user/{user_id}", response_model=FeatureResponse)
async def get_user_features(
    user_id: int,
    feature_names: Optional[List[str]] = Query(default=None),
):
    assert _online_store is not None
    names = feature_names or _default_feature_names("user")

    t0 = time.perf_counter()
    features = _online_store.read_user_features(user_id, names)
    redis_ms = (time.perf_counter() - t0) * 1000
    redis_lookup_latency.labels(entity_type="user").observe(redis_ms)

    missing = _record_cache_misses(features, "user")
    features_served_total.labels(entity_type="user").inc(len(names) - len(missing))

    return FeatureResponse(
        entity_type="user",
        entity_id=user_id,
        features=features,
        missing_features=missing,
        latency_ms=round(redis_ms, 3),
    )


@app.get("/features/item/{item_id}", response_model=FeatureResponse)
async def get_item_features(
    item_id: int,
    feature_names: Optional[List[str]] = Query(default=None),
):
    assert _online_store is not None
    names = feature_names or _default_feature_names("item")

    t0 = time.perf_counter()
    features = _online_store.read_item_features(item_id, names)
    redis_ms = (time.perf_counter() - t0) * 1000
    redis_lookup_latency.labels(entity_type="item").observe(redis_ms)

    missing = _record_cache_misses(features, "item")
    features_served_total.labels(entity_type="item").inc(len(names) - len(missing))

    return FeatureResponse(
        entity_type="item",
        entity_id=item_id,
        features=features,
        missing_features=missing,
        latency_ms=round(redis_ms, 3),
    )


@app.get("/features/vector", response_model=VectorResponse)
async def get_feature_vector(
    user_id: int = Query(...),
    item_id: int = Query(...),
    user_feature_names: Optional[List[str]] = Query(default=None),
    item_feature_names: Optional[List[str]] = Query(default=None),
):
    """Single-pipeline hot path for model serving.  Target: <5 ms."""
    assert _online_store is not None
    u_names = user_feature_names or _default_feature_names("user")
    i_names = item_feature_names or _default_feature_names("item")

    t0 = time.perf_counter()
    features = _online_store.read_feature_vector(user_id, item_id, u_names, i_names)
    redis_ms = (time.perf_counter() - t0) * 1000
    redis_lookup_latency.labels(entity_type="vector").observe(redis_ms)

    missing = [k for k, v in features.items() if v is None]
    for fname in missing:
        entity = "user" if fname.startswith("user_") else "item"
        cache_miss_total.labels(entity_type=entity, feature_name=fname).inc()

    served = len(features) - len(missing)
    features_served_total.labels(entity_type="vector").inc(served)

    return VectorResponse(
        user_id=user_id,
        item_id=item_id,
        features=features,
        missing_features=missing,
        latency_ms=round(redis_ms, 3),
    )


@app.post("/features/batch", response_model=BatchResponse)
async def batch_features(body: BatchRequest):
    assert _online_store is not None
    t_total = time.perf_counter()
    results: List[FeatureResponse] = []

    for req in body.requests:
        t0 = time.perf_counter()
        if req.entity == "user":
            features = _online_store.read_user_features(req.entity_id, req.feature_names)
        elif req.entity == "item":
            features = _online_store.read_item_features(req.entity_id, req.feature_names)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown entity type: {req.entity!r}")

        redis_ms = (time.perf_counter() - t0) * 1000
        missing = _record_cache_misses(features, req.entity)
        features_served_total.labels(entity_type=req.entity).inc(
            len(req.feature_names) - len(missing)
        )
        results.append(
            FeatureResponse(
                entity_type=req.entity,
                entity_id=req.entity_id,
                features=features,
                missing_features=missing,
                latency_ms=round(redis_ms, 3),
            )
        )

    total_ms = (time.perf_counter() - t_total) * 1000
    return BatchResponse(results=results, total_latency_ms=round(total_ms, 3))


@app.get("/health")
async def health():
    assert _online_store is not None
    redis_ok = _online_store.health_check()
    status = "healthy" if redis_ok else "degraded"
    return {"status": status, "redis": redis_ok}


@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/registry", response_model=List[RegistryEntry])
async def registry():
    assert _feature_registry is not None
    return [
        RegistryEntry(
            name=f.name,
            entity=f.entity,
            dtype=f.dtype,
            description=f.description,
            ttl_seconds=f.ttl_seconds,
            window=f.window,
        )
        for f in _feature_registry.list_all()
    ]
