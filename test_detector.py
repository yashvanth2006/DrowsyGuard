import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure the core module is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import config
from core.detector import DrowsyDetector

class TestDrowsyDetector(unittest.TestCase):
    
    @patch('core.detector.mp')
    def setUp(self, mock_mp):
        self.detector = DrowsyDetector()

    def test_extract_eye_region_normal(self):
        # Create a dummy 100x100 frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Dummy landmarks mapping to pixels 40,40 to 60,60
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
        landmarks = [LM(0.4, 0.4), LM(0.6, 0.6)]
        
        # Test extraction
        crop = DrowsyDetector._extract_eye_region(frame, [0, 1], landmarks, 100, 100, padding=5)
        self.assertIsNotNone(crop)
        # Expected shape: min(40)-5 to max(60)+5 => 35 to 65 => 30x30
        self.assertEqual(crop.shape, (30, 30, 3))

    def test_extract_eye_region_out_of_bounds(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
        # Landmarks way outside image boundaries
        landmarks = [LM(1.5, 1.5), LM(1.6, 1.6)]
        
        crop = DrowsyDetector._extract_eye_region(frame, [0, 1], landmarks, 100, 100, padding=10)
        # Should be None if it hits x_min >= x_max (which it will, because max(0, 140) = 140, min(100, 170) = 100 => 140 >= 100)
        self.assertIsNone(crop)

    def test_preprocess_eye_for_cnn_empty(self):
        res = DrowsyDetector._preprocess_eye_for_cnn(None)
        self.assertIsNone(res)
        
        res2 = DrowsyDetector._preprocess_eye_for_cnn(np.array([]))
        self.assertIsNone(res2)

    def test_preprocess_eye_for_cnn_valid(self):
        crop = np.zeros((50, 50, 3), dtype=np.uint8)
        res = DrowsyDetector._preprocess_eye_for_cnn(crop)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, (1, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH, 1))

    @patch('core.detector.mp')
    def test_tensorflow_unavailable(self, mock_mp):
        # Simulate TF import failure
        with patch.dict('sys.modules', {'tensorflow.keras.models': None}):
            detector = DrowsyDetector()
            self.assertFalse(detector.cnn_available)

    def test_hybrid_logic_cnn_unavailable_fallback(self):
        self.detector.cnn_available = False
        self.detector.cnn_model = None
        self.detector.is_calibrated = True
        self.detector.calibrated_threshold = 0.25
        
        # Test EAR only
        self.detector.closed_frames = config.EAR_CONSECUTIVE_FRAMES - 1
        
        with patch.object(self.detector, '_eye_aspect_ratio', return_value=0.20):
            with patch.object(self.detector, '_mouth_aspect_ratio', return_value=0.5):
                # Fake mp_results
                mock_results = MagicMock()
                mock_results.multi_face_landmarks = [MagicMock()]
                self.detector.face_mesh.process.return_value = mock_results
                
                # Should trigger since EAR is < 0.25
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                result = self.detector.process_frame(frame)
                self.assertTrue(result["eyes_closed"])

    def test_hybrid_logic_cnn_available_and_closed(self):
        self.detector.cnn_available = True
        self.detector.cnn_model = MagicMock()
        self.detector.is_calibrated = True
        self.detector.calibrated_threshold = 0.25
        
        # CNN predicts closed (index 1 is 1.0)
        self.detector.cnn_model.predict.return_value = np.array([[0.0, 1.0]])
        
        self.detector.closed_frames = config.EAR_CONSECUTIVE_FRAMES - 1
        
        with patch.object(self.detector, '_eye_aspect_ratio', return_value=0.20):
            with patch.object(self.detector, '_mouth_aspect_ratio', return_value=0.5):
                with patch.object(self.detector, '_extract_eye_region', return_value=np.zeros((10,10,3), dtype=np.uint8)):
                    mock_results = MagicMock()
                    mock_results.multi_face_landmarks = [MagicMock()]
                    self.detector.face_mesh.process.return_value = mock_results
                    
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    result = self.detector.process_frame(frame)
                    
                    self.assertTrue(result["eyes_closed"])
                    self.assertEqual(result["cnn_eye_state"], "CLOSED")

    def test_hybrid_logic_cnn_available_but_open(self):
        self.detector.cnn_available = True
        self.detector.cnn_model = MagicMock()
        self.detector.is_calibrated = True
        self.detector.calibrated_threshold = 0.25
        
        # CNN predicts OPEN (index 1 is 0.0) -> This should suppress the EAR geometric alert
        self.detector.cnn_model.predict.return_value = np.array([[1.0, 0.0]])
        
        self.detector.closed_frames = config.EAR_CONSECUTIVE_FRAMES - 1
        
        with patch.object(self.detector, '_eye_aspect_ratio', return_value=0.20): # Geometric says closed
            with patch.object(self.detector, '_mouth_aspect_ratio', return_value=0.5):
                with patch.object(self.detector, '_extract_eye_region', return_value=np.zeros((10,10,3), dtype=np.uint8)):
                    mock_results = MagicMock()
                    mock_results.multi_face_landmarks = [MagicMock()]
                    self.detector.face_mesh.process.return_value = mock_results
                    
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    result = self.detector.process_frame(frame)
                    
                    # Because CNN strongly disagrees, eyes_closed should be FALSE
                    self.assertFalse(result["eyes_closed"])
                    self.assertEqual(result["cnn_eye_state"], "OPEN")

if __name__ == '__main__':
    unittest.main()
