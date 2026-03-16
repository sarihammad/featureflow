"""Parquet-backed offline feature store.

Storage layout:
    {base_path}/entity={user|item}/date={YYYY-MM-DD}/features.parquet

Each Parquet file is appended (via read-existing + concat + overwrite) so that
multiple writes on the same day accumulate correctly.  The files are partitioned
by entity type and calendar date, which makes point-in-time range queries
efficient when using pandas date filtering.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

ENTITY_ID_COL = "entity_id"
TIMESTAMP_COL = "feature_timestamp"


def _to_utc_timestamp(dt: datetime) -> pd.Timestamp:
    """Convert a datetime (aware or naive) to a UTC-aware pd.Timestamp.

    pandas 2.x raises ValueError if you pass an already-aware datetime
    together with ``tz="UTC"``.  This helper handles both cases correctly.
    """
    if dt.tzinfo is None:
        return pd.Timestamp(dt).tz_localize("UTC")
    return pd.Timestamp(dt).tz_convert("UTC")


@dataclass
class FeatureRecord:
    entity: str
    entity_id: int
    features: Dict[str, Any]
    timestamp: datetime


class OfflineFeatureStore:
    def __init__(self, base_path: str = "data/offline"):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)
        logger.info("OfflineFeatureStore initialised at %s", self._base)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partition_path(self, entity: str, date: str) -> Path:
        partition = self._base / f"entity={entity}" / f"date={date}"
        partition.mkdir(parents=True, exist_ok=True)
        return partition / "features.parquet"

    @staticmethod
    def _flatten_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """Convert non-scalar values (e.g. lists) to JSON strings for Parquet storage."""
        flat: Dict[str, Any] = {}
        for k, v in features.items():
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v)
            elif v is None:
                flat[k] = None
            else:
                flat[k] = v
        return flat

    @staticmethod
    def _unflatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to JSON-decode string values that look like serialised lists/dicts."""
        result: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str) and v.startswith(("[", "{")):
                try:
                    result[k] = json.loads(v)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
            result[k] = v
        return result

    def _read_existing(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Could not read existing parquet at %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write(
        self,
        entity: str,
        entity_id: int,
        features: Dict[str, Any],
        timestamp: datetime,
    ) -> None:
        record = FeatureRecord(
            entity=entity,
            entity_id=entity_id,
            features=features,
            timestamp=timestamp,
        )
        self.write_batch([record])

    def write_batch(self, records: List[FeatureRecord]) -> None:
        if not records:
            return

        # Group by (entity, date) for efficient batch writes
        groups: Dict[tuple, List[FeatureRecord]] = {}
        for r in records:
            key = (r.entity, r.timestamp.strftime("%Y-%m-%d"))
            groups.setdefault(key, []).append(r)

        for (entity, date), recs in groups.items():
            path = self._partition_path(entity, date)
            rows = []
            for r in recs:
                row = {
                    ENTITY_ID_COL: r.entity_id,
                    TIMESTAMP_COL: r.timestamp,
                }
                row.update(self._flatten_features(r.features))
                rows.append(row)

            new_df = pd.DataFrame(rows)
            existing_df = self._read_existing(path)

            if existing_df is not None:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined = new_df

            combined[TIMESTAMP_COL] = pd.to_datetime(combined[TIMESTAMP_COL], utc=True)
            combined.sort_values(TIMESTAMP_COL, inplace=True)
            combined.reset_index(drop=True, inplace=True)

            table = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(table, str(path), compression="snappy")

        logger.debug("Wrote %d records to offline store", len(records))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def read_entity_history(
        self,
        entity: str,
        entity_id: int,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Return all feature snapshots for a single entity within [start_time, end_time)."""
        entity_dir = self._base / f"entity={entity}"
        if not entity_dir.exists():
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for date_dir in sorted(entity_dir.iterdir()):
            parquet_file = date_dir / "features.parquet"
            if not parquet_file.exists():
                continue
            try:
                df = pd.read_parquet(str(parquet_file))
            except Exception as exc:
                logger.warning("Skipping unreadable parquet %s: %s", parquet_file, exc)
                continue

            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
            mask = (
                (df[ENTITY_ID_COL] == entity_id)
                & (df[TIMESTAMP_COL] >= _to_utc_timestamp(start_time))
                & (df[TIMESTAMP_COL] < _to_utc_timestamp(end_time))
            )
            filtered = df[mask]
            if not filtered.empty:
                frames.append(filtered)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.sort_values(TIMESTAMP_COL, inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def read_all_latest(
        self,
        entity: str,
        as_of: datetime,
    ) -> pd.DataFrame:
        """Return the most recent feature snapshot per entity as of *as_of*.

        This is the building block for point-in-time joins: each entity
        contributes exactly one row — its last-known state before the
        reference timestamp.
        """
        entity_dir = self._base / f"entity={entity}"
        if not entity_dir.exists():
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        as_of_ts = _to_utc_timestamp(as_of)

        for date_dir in sorted(entity_dir.iterdir()):
            parquet_file = date_dir / "features.parquet"
            if not parquet_file.exists():
                continue
            try:
                df = pd.read_parquet(str(parquet_file))
            except Exception as exc:
                logger.warning("Skipping unreadable parquet %s: %s", parquet_file, exc)
                continue

            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
            mask = df[TIMESTAMP_COL] < as_of_ts
            filtered = df[mask]
            if not filtered.empty:
                frames.append(filtered)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(TIMESTAMP_COL, inplace=True)
        # Keep only the last snapshot per entity
        latest = combined.groupby(ENTITY_ID_COL, as_index=False).last()
        latest.reset_index(drop=True, inplace=True)
        return latest
