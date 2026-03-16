import logging
from typing import Callable, List, Optional

from kafka import KafkaConsumer as _KafkaConsumer
from kafka.errors import KafkaError

from src.events.schema import UserEvent, UserEventModel

logger = logging.getLogger(__name__)


class EventConsumer:
    """Kafka consumer that deserializes raw messages into UserEvent objects.

    Designed for at-least-once delivery: offsets are committed only after the
    caller's callback has returned successfully.  Deserialization errors are
    logged and skipped so a single malformed message cannot stall the pipeline.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "user-events",
        group_id: str = "feature-pipeline",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = False,
        session_timeout_ms: int = 30_000,
        max_poll_records: int = 500,
    ):
        self.topic = topic
        self._consumer = _KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            value_deserializer=lambda v: v.decode("utf-8", errors="replace"),
            session_timeout_ms=session_timeout_ms,
            max_poll_records=max_poll_records,
        )
        logger.info(
            "EventConsumer connected to %s, topic=%s, group=%s",
            bootstrap_servers,
            topic,
            group_id,
        )

    def consume(
        self,
        batch_size: int = 100,
        timeout_ms: int = 1000,
    ) -> List[UserEvent]:
        raw_batch = self._consumer.poll(timeout_ms=timeout_ms, max_records=batch_size)
        events: List[UserEvent] = []
        for _tp, messages in raw_batch.items():
            for msg in messages:
                event = self._deserialize(msg.value)
                if event is not None:
                    events.append(event)
        return events

    def consume_forever(
        self,
        callback: Callable[[List[UserEvent]], None],
        batch_size: int = 100,
        timeout_ms: int = 1000,
    ) -> None:
        logger.info("Starting consume_forever loop on topic=%s", self.topic)
        while True:
            events = self.consume(batch_size=batch_size, timeout_ms=timeout_ms)
            if events:
                try:
                    callback(events)
                except Exception as exc:
                    logger.error("Callback raised an exception: %s", exc, exc_info=True)
                else:
                    self._consumer.commit()

    def close(self) -> None:
        self._consumer.close()

    @staticmethod
    def _deserialize(raw: str) -> Optional[UserEvent]:
        try:
            model = UserEventModel.from_json(raw)
            return model.to_event()
        except Exception as exc:
            logger.warning("Failed to deserialize event (skipping): %s | raw=%s", exc, raw[:200])
            return None
