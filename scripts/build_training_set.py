#!/usr/bin/env python3
"""CLI: build a point-in-time correct training dataset from historical events.

This script:
  1. Generates a configurable volume of historical user events via the
     EventGenerator (no external dataset required).
  2. Runs the BatchProcessor to materialise hourly feature snapshots to the
     offline store.
  3. Derives a label dataset from purchase events (label=1) and sampled
     non-purchase item views (label=0).
  4. Runs the PointInTimeDatasetBuilder to produce a leakage-free training CSV.
  5. Prints a leakage validation report.

Usage
-----
    python scripts/build_training_set.py
    python scripts/build_training_set.py --n-events 100000 --output data/training.csv
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.events.generator import EventGenerator
from src.events.schema import EventType
from src.features.registry import _registry
from src.pipeline.batch_processor import BatchProcessor
from src.stores.offline_store import OfflineFeatureStore
from src.training.dataset_builder import PointInTimeDatasetBuilder

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_training_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PIT-correct training dataset")
    parser.add_argument("--n-events", type=int, default=50_000, help="Number of historical events to generate")
    parser.add_argument("--span-hours", type=float, default=48.0, help="Time span of historical events in hours")
    parser.add_argument("--offline-path", default=settings.OFFLINE_STORE_PATH)
    parser.add_argument("--output", default="data/training_dataset.csv")
    parser.add_argument("--n-users", type=int, default=1000, help="Subset of users for faster demo")
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freq", default="1H", choices=["1H", "6H", "12H", "1D"])
    return parser.parse_args()


def build_label_events(events, max_labels: int = 5000) -> pd.DataFrame:
    """Derive binary labels: purchase=1, item_view (non-purchased)=0."""
    purchased_keys = set()
    label_rows = []

    for e in events:
        if e.event_type == EventType.PURCHASE and e.item_id is not None:
            purchased_keys.add((e.user_id, e.item_id))
            label_rows.append({
                "user_id": e.user_id,
                "item_id": e.item_id,
                "timestamp": e.timestamp,
                "label": 1,
            })

    for e in events:
        if (
            e.event_type == EventType.ITEM_VIEW
            and e.item_id is not None
            and (e.user_id, e.item_id) not in purchased_keys
        ):
            label_rows.append({
                "user_id": e.user_id,
                "item_id": e.item_id,
                "timestamp": e.timestamp,
                "label": 0,
            })
            if len(label_rows) >= max_labels:
                break

    df = pd.DataFrame(label_rows)
    df.drop_duplicates(subset=["user_id", "item_id", "label"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate historical events
    # ------------------------------------------------------------------
    logger.info("Generating %d historical events over %.0f hours ...", args.n_events, args.span_hours)
    generator = EventGenerator(
        n_users=args.n_users,
        n_items=args.n_items,
        seed=args.seed,
    )
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=args.span_hours)
    events = generator.generate_historical(
        n_events=args.n_events,
        end_time=end_time,
        span_hours=args.span_hours,
    )
    logger.info("Generated %d events between %s and %s", len(events), start_time.isoformat(), end_time.isoformat())

    # ------------------------------------------------------------------
    # Step 2: Materialise feature snapshots to offline store
    # ------------------------------------------------------------------
    offline_store = OfflineFeatureStore(base_path=args.offline_path)
    batch_proc = BatchProcessor(offline_store=offline_store, registry=_registry)

    logger.info("Materialising hourly feature snapshots (freq=%s) ...", args.freq)
    batch_proc.materialize_date_range(
        events=events,
        start_date=start_time + timedelta(hours=1),
        end_date=end_time,
        freq=args.freq,
    )
    logger.info("Feature materialisation complete.")

    # ------------------------------------------------------------------
    # Step 3: Build label events
    # ------------------------------------------------------------------
    label_df = build_label_events(events)
    logger.info(
        "Label events: %d rows (%d positive, %d negative)",
        len(label_df),
        (label_df["label"] == 1).sum(),
        (label_df["label"] == 0).sum(),
    )

    if label_df.empty:
        logger.error("No label events found. Try increasing --n-events.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Point-in-time correct join
    # ------------------------------------------------------------------
    builder = PointInTimeDatasetBuilder(offline_store=offline_store, registry=_registry)

    user_feature_names = [f.name for f in _registry.list_user_features()]
    item_feature_names = [f.name for f in _registry.list_item_features()]

    logger.info("Running point-in-time join ...")
    dataset = builder.build(
        label_events=label_df,
        user_features=user_feature_names,
        item_features=item_feature_names,
    )

    if dataset.empty:
        logger.error(
            "Training dataset is empty after PIT join. "
            "This usually means the offline store has no snapshots before the label timestamps. "
            "Try increasing --span-hours or lowering --freq."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Validate no leakage
    # ------------------------------------------------------------------
    report = builder.validate_no_leakage(dataset, label_df)
    logger.info(str(report))
    if not report.passed:
        logger.error(
            "LEAKAGE DETECTED: %d violations. Details: %s",
            report.violations,
            report.details[:5],
        )

    # ------------------------------------------------------------------
    # Step 6: Save
    # ------------------------------------------------------------------
    dataset.to_csv(str(output_path), index=False)
    logger.info(
        "Saved training dataset to %s (%d rows x %d columns)",
        output_path,
        len(dataset),
        len(dataset.columns),
    )
    print(f"\nTraining dataset preview:\n{dataset.head(5).to_string()}\n")
    print(f"Columns: {list(dataset.columns)}")
    print(f"Shape: {dataset.shape}")
    print(f"Leakage check: {report}")


if __name__ == "__main__":
    main()
