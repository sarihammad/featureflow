import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class EventType(str, Enum):
    PAGE_VIEW = "page_view"
    ITEM_VIEW = "item_view"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"
    SEARCH = "search"
    RATING = "rating"


@dataclass
class UserEvent:
    event_id: str
    user_id: int
    event_type: EventType
    item_id: Optional[int]
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "item_id": self.item_id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


class UserEventModel(BaseModel):
    event_id: str
    user_id: int
    event_type: EventType
    item_id: Optional[int] = None
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any] = {}

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "UserEventModel":
        return cls.model_validate_json(data)

    def to_event(self) -> UserEvent:
        return UserEvent(
            event_id=self.event_id,
            user_id=self.user_id,
            event_type=self.event_type,
            item_id=self.item_id,
            timestamp=self.timestamp,
            session_id=self.session_id,
            metadata=self.metadata,
        )

    @classmethod
    def from_event(cls, event: UserEvent) -> "UserEventModel":
        return cls(
            event_id=event.event_id,
            user_id=event.user_id,
            event_type=event.event_type,
            item_id=event.item_id,
            timestamp=event.timestamp,
            session_id=event.session_id,
            metadata=event.metadata,
        )
