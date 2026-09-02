import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from unittest.mock import patch

import config
from analytics.logger import SessionLogger
from analytics.session import Session, Event

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for local analytics dir
        self.test_dir = tempfile.mkdtemp()
        self.patcher = patch('config.LOCAL_ANALYTICS_DIR', new=self.test_dir)
        self.patcher.start()
        
        self.logger = SessionLogger()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_session_creation(self):
        session_id = self.logger.start_session()
        self.assertIsNotNone(session_id)
        self.assertTrue(len(session_id) > 0)
        self.assertIsNotNone(self.logger.session)
        self.assertEqual(self.logger.session.session_id, session_id)
        
        # Verify SESSION_STARTED event is logged
        self.assertEqual(len(self.logger.session.events), 1)
        self.assertEqual(self.logger.session.events[0]["event_type"], "SESSION_STARTED")

    def test_uuid_uniqueness(self):
        id1 = self.logger.start_session()
        self.logger.end_session()
        
        logger2 = SessionLogger()
        id2 = logger2.start_session()
        
        self.assertNotEqual(id1, id2)

    def test_metrics_updates(self):
        self.logger.start_session()
        
        result = {
            "is_calibrated": True,
            "baseline_ear": 0.3,
            "eyes_closed": False,
            "ear": 0.28,
            "cnn_available": True,
            "cnn_eye_state": "OPEN"
        }
        
        self.logger.update_metrics(result, "LOW", False)
        
        self.assertTrue(self.logger.session.calibration_completed)
        self.assertEqual(self.logger.session.maximum_risk_level, "LOW")
        self.assertTrue(self.logger.session.cnn_available)
        self.assertEqual(self.logger.session.total_drowsy_events, 0)
        self.assertEqual(self.logger.session.total_alerts, 0)
        self.assertEqual(self.logger.session.average_ear, 0.28)

    def test_drowsiness_deduplication(self):
        self.logger.start_session()
        
        # Frame 1: Drowsy
        self.logger.update_metrics({"eyes_closed": True, "ear": 0.15}, "HIGH", True)
        self.assertTrue(self.logger.active_drowsy_episode)
        self.assertEqual(self.logger.session.total_drowsy_events, 1)
        
        # Frame 2: Still Drowsy
        self.logger.update_metrics({"eyes_closed": True, "ear": 0.15}, "HIGH", False)
        self.assertEqual(self.logger.session.total_drowsy_events, 1) # Should not increment
        
        # Frame 3: Normal
        self.logger.update_metrics({"eyes_closed": False, "ear": 0.3}, "LOW", False)
        self.assertFalse(self.logger.active_drowsy_episode)
        self.assertEqual(self.logger.session.total_drowsy_events, 1)
        
        events = [e["event_type"] for e in self.logger.session.events]
        self.assertIn("DROWSINESS_DETECTED", events)
        self.assertIn("DROWSINESS_CLEARED", events)

    def test_risk_transitions(self):
        self.logger.start_session()
        
        self.logger.update_metrics({"ear": 0.3}, "LOW", False)
        self.logger.update_metrics({"ear": 0.3}, "LOW", False)
        self.logger.update_metrics({"ear": 0.2}, "MEDIUM", False)
        self.logger.update_metrics({"ear": 0.2}, "MEDIUM", False)
        self.logger.update_metrics({"ear": 0.1}, "HIGH", False)
        
        events = [e for e in self.logger.session.events if e["event_type"] == "RISK_LEVEL_CHANGED"]
        
        # LOW -> MEDIUM and MEDIUM -> HIGH
        self.assertEqual(len(events), 2)
        self.assertEqual(self.logger.session.maximum_risk_level, "HIGH")

    def test_persistence_atomic_storage(self):
        session_id = self.logger.start_session()
        self.logger.update_metrics({"ear": 0.3}, "LOW", False)
        self.logger.end_session()
        
        file_path = Path(self.test_dir) / f"{session_id}.json"
        self.assertTrue(file_path.exists())
        
        with open(file_path, "r") as f:
            data = json.load(f)
            
        self.assertEqual(data["session_id"], session_id)
        self.assertEqual(data["maximum_risk_level"], "LOW")
        self.assertIsNotNone(data["end_time"])
        self.assertIn("duration", data)

    @patch('analytics.logger.Path.replace')
    def test_disk_failure(self, mock_replace):
        mock_replace.side_effect = PermissionError("Mock write failure")
        
        self.logger.start_session()
        self.logger.end_session() # Should catch the error and not crash
        
        # Application should survive

    def test_session_isolation(self):
        logger1 = SessionLogger()
        id1 = logger1.start_session()
        logger1.update_metrics({"ear": 0.3}, "LOW", False)
        
        logger2 = SessionLogger()
        id2 = logger2.start_session()
        logger2.update_metrics({"ear": 0.1}, "HIGH", True)
        
        self.assertNotEqual(logger1.session.total_alerts, logger2.session.total_alerts)
        self.assertEqual(logger1.session.maximum_risk_level, "LOW")
        self.assertEqual(logger2.session.maximum_risk_level, "HIGH")

    def test_cloud_disabled(self):
        self.assertFalse(config.CLOUD_ANALYTICS_ENABLED)
        # There's no cloud logic in logger to test yet, but we verify it's disabled.

if __name__ == '__main__':
    unittest.main()
