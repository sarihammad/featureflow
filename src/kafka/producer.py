import logging
from typing import List, Optional

from kafka import KafkaProducer as _KafkaProducer
from kafka.errors import KafkaError

from src.events.schema import UserEvent, UserEventModel

logger = logging.getLogger(__name__)


class EventProducer:
    """Kafka producer that publishes UserEvent objects to a configured topic.

    Events are keyed by user_id so that all events for a given user land on
    the same partition, guaranteeing per-user ordering inside the consumer
    without any cross-partition coordination.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "user-events",
        acks: str = "all",
        retries: int = 3,
        linger_ms: int = 5,
    ):
        self.topic = topic
        self._producer = _KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            key_serializer=lambda k: str(k).encode("utf-8"),
            value_serializer=lambda v: v.encode("utf-8"),
            acks=acks,
            retries=retries,
            linger_ms=linger_ms,
            compression_type="gzip",
        )
        logger.info("EventProducer connected to %s, topic=%s", bootstrap_servers, topic)

    def produce(self, event: UserEvent) -> None:
        model = UserEventModel.from_event(event)
        payload = model.to_json()
        future = self._producer.send(
            self.topic,
            key=event.user_id,
            value=payload,
        )
        future.add_errback(self._on_error)

    def produce_batch(self, events: List[UserEvent]) -> None:
        for event in events:
            self.produce(event)
        self._producer.poll(timeout_ms=0)

    def flush(self, timeout: Optional[float] = 30.0) -> None:
        self._producer.flush(timeout=timeout)

    def close(self) -> None:
        self._producer.close()

    @staticmethod
    def _on_error(exc: KafkaError) -> None:
        logger.error("Kafka delivery failure: %s", exc)
