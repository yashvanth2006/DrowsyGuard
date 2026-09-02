import json
import logging
import os
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import config
from analytics.session import Session, Event

logger = logging.getLogger(__name__)

class SessionLogger:
    def __init__(self):
        self.session = None
        self.active_drowsy_episode = False
        self.active_yawn_episode = False
        
        # Risk level tracking
        self.risk_levels = {"Unknown": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        self.current_risk = "Unknown"
        
        # EAR tracking
        self.ear_sum = 0.0
        self.ear_count = 0
        
        if config.ANALYTICS_ENABLED:
            os.makedirs(config.LOCAL_ANALYTICS_DIR, exist_ok=True)
            
    def start_session(self) -> str:
        if not config.ANALYTICS_ENABLED:
            return ""
            
        self.session = Session.create()
        self.active_drowsy_episode = False
        self.active_yawn_episode = False
        self.current_risk = "Unknown"
        self.ear_sum = 0.0
        self.ear_count = 0
        
        self.log_event("SESSION_STARTED")
        return self.session.session_id
        
    def log_event(self, event_type: str, metadata: dict = None):
        if not self.session or not config.ANALYTICS_ENABLED:
            return
            
        try:
            event = Event.create(self.session.session_id, event_type, metadata)
            self.session.events.append(event.to_dict())
        except Exception as e:
            logger.error(f"Analytics event logging failed: {e}")
            
    def update_metrics(self, result: dict, risk_level: str, alert_triggered: bool):
        if not self.session or not config.ANALYTICS_ENABLED:
            return
            
        try:
            # Update Calibration status
            if result.get("is_calibrated") and not self.session.calibration_completed:
                self.session.calibration_completed = True
                self.log_event("CALIBRATION_COMPLETED", {"baseline_ear": result.get("baseline_ear")})
            elif result.get("state") == "CALIBRATING":
                # Ensure we capture calibration started if possible, though app_final.py logic is simple
                if len(self.session.events) == 1: # Only SESSION_STARTED exists
                    self.log_event("CALIBRATION_STARTED")
            
            # Risk Level Transitions
            if risk_level != self.current_risk:
                if self.current_risk != "Unknown":
                    self.log_event("RISK_LEVEL_CHANGED", {"from": self.current_risk, "to": risk_level})
                self.current_risk = risk_level
                
                # Update Max Risk
                if self.risk_levels.get(risk_level, 0) > self.risk_levels.get(self.session.maximum_risk_level, 0):
                    self.session.maximum_risk_level = risk_level
            
            # Drowsiness Deduplication
            is_drowsy = result.get("eyes_closed", False)
            if is_drowsy and not self.active_drowsy_episode:
                self.active_drowsy_episode = True
                self.session.total_drowsy_events += 1
                metadata = {
                    "ear": result.get("ear"),
                    "cnn_eye_state": result.get("cnn_eye_state"),
                    "risk_level": risk_level
                }
                self.log_event("DROWSINESS_DETECTED", metadata)
            elif not is_drowsy and self.active_drowsy_episode:
                self.active_drowsy_episode = False
                self.log_event("DROWSINESS_CLEARED")
                
            # Yawn Deduplication
            is_yawning = result.get("yawning", False)
            if is_yawning and not self.active_yawn_episode:
                self.active_yawn_episode = True
                self.session.total_yawns += 1
                self.log_event("YAWN_DETECTED", {"mar": result.get("mar")})
            elif not is_yawning and self.active_yawn_episode:
                self.active_yawn_episode = False
                self.log_event("YAWN_CLEARED")

            # Alert triggering
            if alert_triggered:
                self.session.total_alerts += 1
                self.log_event("ALERT_TRIGGERED", {"risk_level": risk_level, "reason": "drowsiness" if is_drowsy else ("yawn" if is_yawning else "unknown")})

            # Continuous metrics update
            ear = result.get("ear", 0.0)
            if ear > 0:
                self.ear_sum += ear
                self.ear_count += 1
                self.session.average_ear = self.ear_sum / self.ear_count
                if ear < self.session.minimum_ear:
                    self.session.minimum_ear = ear

            # CNN metrics
            if result.get("cnn_available"):
                self.session.cnn_available = True
                if result.get("cnn_eye_state") == "CLOSED":
                    self.session.cnn_closed_eye_detections += 1

        except Exception as e:
            logger.error(f"Analytics metric update failed: {e}")

    def end_session(self):
        if not self.session or not config.ANALYTICS_ENABLED:
            return
            
        try:
            self.log_event("SESSION_STOPPED")
            
            end_time = datetime.utcnow()
            self.session.end_time = end_time.isoformat() + "Z"
            
            start_time = datetime.fromisoformat(self.session.start_time.replace("Z", ""))
            self.session.duration = int((end_time - start_time).total_seconds())
            
            if self.active_drowsy_episode:
                self.log_event("DROWSINESS_CLEARED")
                
            if self.active_yawn_episode:
                self.log_event("YAWN_CLEARED")

            # Save to disk using atomic write
            self._save_session_to_disk()
            
        except Exception as e:
            logger.error(f"Failed to end and save session: {e}")
        finally:
            self.session = None
            
    def _save_session_to_disk(self):
        if not self.session:
            return
            
        try:
            target_path = Path(config.LOCAL_ANALYTICS_DIR) / f"{self.session.session_id}.json"
            temp_path = target_path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.session.to_dict(), f, indent=2)
                
            # Atomic rename (replace)
            temp_path.replace(target_path)
            
        except Exception as e:
            logger.error(f"Failed to save session data to disk: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

    def get_session_history(self) -> list:
        if not config.ANALYTICS_ENABLED:
            return []
            
        history = []
        try:
            dir_path = Path(config.LOCAL_ANALYTICS_DIR)
            if not dir_path.exists():
                return []
                
            for file_path in dir_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # We only want a summary
                        history.append({
                            "session_id": data.get("session_id"),
                            "date": data.get("start_time", "").split("T")[0] + " " + data.get("start_time", "T00:00").split("T")[1][:5],
                            "duration": data.get("duration", 0) // 60,  # minutes
                            "alerts": data.get("total_alerts", 0),
                            "risk": data.get("maximum_risk_level", "Unknown"),
                            "start_time_raw": data.get("start_time")
                        })
                except Exception as e:
                    logger.warning(f"Failed to read session file {file_path}: {e}")
                    
            # Sort by raw start time descending
            history.sort(key=lambda x: x.get("start_time_raw", ""), reverse=True)
            return history
        except Exception as e:
            logger.error(f"Failed to load session history: {e}")
            return []
