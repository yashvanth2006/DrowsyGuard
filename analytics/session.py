from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

@dataclass
class Event:
    event_id: str
    session_id: str
    event_type: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create(cls, session_id: str, event_type: str, metadata: Optional[Dict[str, Any]] = None):
        return cls(
            event_id=uuid.uuid4().hex,
            session_id=session_id,
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            metadata=metadata or {}
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Session:
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    duration: int = 0  # seconds
    calibration_completed: bool = False
    total_alerts: int = 0
    maximum_risk_level: str = "Unknown"
    total_drowsy_events: int = 0
    total_yawns: int = 0
    average_ear: float = 0.0
    minimum_ear: float = 1.0
    cnn_available: bool = False
    cnn_closed_eye_detections: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls):
        return cls(
            session_id=uuid.uuid4().hex,
            start_time=datetime.utcnow().isoformat() + "Z"
        )

    def to_dict(self):
        return asdict(self)
