#!/usr/bin/env python3
"""CLI: run the stream processor (consume Kafka → compute features → write stores).

Usage
-----
    python scripts/run_processor.py
    python scripts/run_processor.py --offline-path data/offline --log-level DEBUG
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.features.registry import _registry
from src.kafka.consumer import EventConsumer
from src.pipeline.stream_processor import StreamProcessor
from src.stores.offline_store import OfflineFeatureStore
from src.stores.online_store import OnlineFeatureStore

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_processor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FeatureFlow stream processor")
    parser.add_argument("--bootstrap-servers", default=settings.KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=settings.KAFKA_TOPIC_USER_EVENTS)
    parser.add_argument("--group-id", default=settings.KAFKA_CONSUMER_GROUP)
    parser.add_argument("--redis-url", default=settings.REDIS_URL)
    parser.add_argument("--offline-path", default=settings.OFFLINE_STORE_PATH)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--auto-offset-reset", default="latest", choices=["latest", "earliest"])
    parser.add_argument("--log-level", default=settings.LOG_LEVEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    consumer = EventConsumer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        auto_offset_reset=args.auto_offset_reset,
    )

    online_store = OnlineFeatureStore(
        redis_url=args.redis_url,
        default_ttl=settings.FEATURE_TTL_SECONDS,
    )

    offline_store = OfflineFeatureStore(base_path=args.offline_path)

    processor = StreamProcessor(
        consumer=consumer,
        online_store=online_store,
        offline_store=offline_store,
        registry=_registry,
    )

    logger.info("StreamProcessor ready. Waiting for events on topic=%s ...", args.topic)
    try:
        processor.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down stream processor.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
