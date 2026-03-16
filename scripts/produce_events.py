#!/usr/bin/env python3
"""CLI: generate synthetic user events and publish them to Kafka.

Usage
-----
    python scripts/produce_events.py --n-events 10000
    python scripts/produce_events.py --n-events 50000 --events-per-second 500
    python scripts/produce_events.py --stream --duration 60  # stream for 60s
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.events.generator import EventGenerator
from src.kafka.producer import EventProducer

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("produce_events")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish synthetic user events to Kafka")
    parser.add_argument("--n-events", type=int, default=10_000, help="Number of events to produce (batch mode)")
    parser.add_argument("--events-per-second", type=float, default=settings.EVENTS_PER_SECOND)
    parser.add_argument("--n-users", type=int, default=settings.N_USERS)
    parser.add_argument("--n-items", type=int, default=settings.N_ITEMS)
    parser.add_argument("--bootstrap-servers", default=settings.KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=settings.KAFKA_TOPIC_USER_EVENTS)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--stream", action="store_true", help="Stream events continuously")
    parser.add_argument("--duration", type=float, default=None, help="Streaming duration in seconds")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generator = EventGenerator(
        n_users=args.n_users,
        n_items=args.n_items,
        events_per_second=args.events_per_second,
        seed=args.seed,
    )

    producer = EventProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    )

    try:
        if args.stream:
            logger.info(
                "Streaming events at %.0f eps to topic=%s (duration=%s)",
                args.events_per_second,
                args.topic,
                f"{args.duration}s" if args.duration else "infinite",
            )
            produced = 0
            t0 = time.monotonic()
            for event in generator.stream(duration_seconds=args.duration):
                producer.produce(event)
                produced += 1
                if produced % 1000 == 0:
                    elapsed = time.monotonic() - t0
                    logger.info("Streamed %d events in %.1fs", produced, elapsed)
        else:
            logger.info(
                "Producing %d events in batches of %d to topic=%s",
                args.n_events,
                args.batch_size,
                args.topic,
            )
            produced = 0
            t0 = time.monotonic()

            while produced < args.n_events:
                batch_n = min(args.batch_size, args.n_events - produced)
                batch = generator.generate_batch(batch_n)
                producer.produce_batch(batch)
                produced += batch_n

                if produced % 2000 == 0 or produced == args.n_events:
                    elapsed = time.monotonic() - t0
                    rate = produced / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Produced %d/%d events (%.0f/s)",
                        produced,
                        args.n_events,
                        rate,
                    )

        producer.flush()
        logger.info("Done. Flushed all messages.")

    except KeyboardInterrupt:
        logger.info("Interrupted — flushing and exiting.")
        producer.flush(timeout=5.0)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
