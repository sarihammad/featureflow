"""Prometheus instrumentation middleware for the FeatureFlow serving API."""

import time

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

feature_request_latency = Histogram(
    "feature_request_latency_ms",
    "End-to-end request latency in milliseconds",
    ["entity_type", "endpoint"],
    buckets=[0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
)

redis_lookup_latency = Histogram(
    "redis_lookup_latency_ms",
    "Redis pipeline lookup latency in milliseconds",
    ["entity_type"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50],
)

cache_miss_total = Counter(
    "cache_miss_total",
    "Number of feature keys requested but not present in Redis",
    ["entity_type", "feature_name"],
)

features_served_total = Counter(
    "features_served_total",
    "Total number of feature values returned by the serving API",
    ["entity_type"],
)

active_connections = Gauge(
    "active_connections",
    "Number of currently active HTTP connections",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        active_connections.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            active_connections.dec()

            # Determine entity type from path for label cardinality control
            path = request.url.path
            if "/user/" in path:
                entity_type = "user"
            elif "/item/" in path:
                entity_type = "item"
            elif "/vector" in path:
                entity_type = "vector"
            else:
                entity_type = "other"

            feature_request_latency.labels(
                entity_type=entity_type,
                endpoint=path,
            ).observe(elapsed_ms)

        return response
