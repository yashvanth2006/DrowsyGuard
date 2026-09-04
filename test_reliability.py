import unittest
from unittest.mock import patch, MagicMock
import sys
import time
import numpy as np

# We want to mock time.sleep so our tests run instantly instead of waiting for 0.1s during failure loops
patch('time.sleep').start()

import drowsy_detect

class TestReliability(unittest.TestCase):
    def setUp(self):
        drowsy_detect.last_alert = 0

    @patch('cv2.VideoCapture')
    @patch('drowsy_detect.DrowsyDetector')
    @patch('pygame.mixer.quit')
    def test_camera_open_failure(self, mock_mixer_quit, mock_detector_cls, mock_videocapture):
        # Mock cap.isOpened() to False
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap
        
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance

        with self.assertRaises(SystemExit):
            drowsy_detect.main()
            
        mock_detector_instance.close.assert_called_once()
        mock_mixer_quit.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('cv2.destroyAllWindows')
    @patch('drowsy_detect.DrowsyDetector')
    def test_camera_persistent_read_failure(self, mock_detector_cls, mock_destroy, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None) # Always fail
        mock_videocapture.return_value = mock_cap
        
        mock_detector_instance = MagicMock()
        mock_detector_cls.return_value = mock_detector_instance

        # Should break after 5 consecutive failures
        drowsy_detect.main()
        
        self.assertEqual(mock_cap.read.call_count, 5)
        mock_cap.release.assert_called_once()
        mock_detector_instance.close.assert_called_once()
        mock_destroy.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('cv2.waitKey')
    @patch('cv2.imshow')
    @patch('drowsy_detect.DrowsyDetector')
    def test_temporary_read_failure_resets_counter(self, mock_detector_cls, mock_imshow, mock_waitkey, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        import numpy as np
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Sequence: Fail(1), Fail(2), Success(resets to 0), Fail(1), Fail(2), Fail(3), Fail(4), Success(resets to 0)
        reads = [
            (False, None), (False, None), 
            (True, fake_frame), 
            (False, None), (False, None), (False, None), (False, None), 
            (True, fake_frame)
        ]
        
        def read_side_effect():
            if reads:
                return reads.pop(0)
            return (True, fake_frame)
        
        mock_cap.read.side_effect = read_side_effect
        mock_videocapture.return_value = mock_cap
        
        def waitkey_side_effect(delay):
            # Quit when we are out of pre-defined reads
            if len(reads) == 0:
                return ord('q')
            return -1
            
        mock_waitkey.side_effect = waitkey_side_effect

        mock_detector_instance = MagicMock()
        mock_detector_instance.process_frame.return_value = {"face_detected": False}
        mock_detector_cls.return_value = mock_detector_instance

        # Main should complete successfully via 'q' key instead of breaking due to 5 failures
        drowsy_detect.main()
        
        # Ensure cleanup was still done
        mock_cap.release.assert_called_once()
        mock_detector_instance.close.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('cv2.destroyAllWindows')
    @patch('drowsy_detect.DrowsyDetector')
    def test_detector_exception_cleanup(self, mock_detector_cls, mock_destroy, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_videocapture.return_value = mock_cap
        
        mock_detector_instance = MagicMock()
        mock_detector_instance.process_frame.side_effect = Exception("Simulated Detector Error")
        mock_detector_cls.return_value = mock_detector_instance

        # Main should catch exception and run finally block
        drowsy_detect.main()
        
        mock_cap.release.assert_called_once()
        mock_detector_instance.close.assert_called_once()
        mock_destroy.assert_called_once()

    def test_no_streamlit_dependency(self):
        with open('drowsy_detect.py', 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertNotIn("import streamlit", content)
            self.assertNotIn("from streamlit", content)

if __name__ == '__main__':
    unittest.main()
