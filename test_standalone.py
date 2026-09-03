import unittest
from unittest.mock import patch, MagicMock
import sys
import os

class TestStandalone(unittest.TestCase):
    
    def test_detector_import(self):
        """Test 1: Verify that DrowsyDetector can be imported."""
        try:
            from core.detector import DrowsyDetector
        except Exception as e:
            self.fail(f"Could not import DrowsyDetector: {e}")

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    def test_standalone_import_does_not_execute(self, mock_imshow, mock_video_capture):
        """Test 2: Verify that importing drowsy_detect does not start the camera or GUI loop."""
        import drowsy_detect
        mock_video_capture.assert_not_called()
        mock_imshow.assert_not_called()

    @patch('drowsy_detect.cv2')
    @patch('drowsy_detect.DrowsyDetector')
    def test_detector_delegation(self, mock_detector_class, mock_cv2):
        """Test 3: Verify that drowsy_detect.main() uses DrowsyDetector."""
        import drowsy_detect
        
        # Setup mocks
        mock_detector_instance = MagicMock()
        mock_detector_class.return_value = mock_detector_instance
        
        # Mock frame and result
        mock_cv2.VideoCapture().read.side_effect = [(True, "frame"), (False, None)]
        mock_cv2.waitKey.return_value = ord('q')
        
        # We need to mock process_frame to return a valid result dictionary
        mock_detector_instance.process_frame.return_value = {
            "face_detected": True,
            "ear": 0.3,
            "mar": 0.5,
            "state": "NORMAL",
            "cnn_available": False,
            "is_calibrated": False
        }
        
        # Run main
        drowsy_detect.main()
        
        # Assert DrowsyDetector was instantiated
        mock_detector_class.assert_called_once()
        # Assert process_frame was called
        mock_detector_instance.process_frame.assert_called()
        # Assert close was called
        mock_detector_instance.close.assert_called_once()

    def test_no_streamlit_dependency(self):
        """Test 4: Verify that drowsy_detect.py does not require streamlit."""
        with open('drowsy_detect.py', 'r') as f:
            content = f.read()
        self.assertNotIn('import streamlit', content)
        self.assertNotIn('st.', content)

if __name__ == '__main__':
    unittest.main()
