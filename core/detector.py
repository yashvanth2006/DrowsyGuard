import logging
import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp

import config

logger = logging.getLogger(__name__)

class DrowsyDetector:
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 39, 269, 405, 291, 375, 321, 308]

    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        # CNN initialization
        self.cnn_model = None
        self.cnn_available = False
        self._load_cnn()

        # Calibration state
        self.is_calibrated = False
        self.is_calibrating = False
        self.calibration_samples = []
        self.baseline_ear = config.DEFAULT_BASELINE_EAR
        self.calibrated_threshold = config.STATIC_EAR_THRESHOLD

        # Detection state
        self.closed_frames = 0
        self.yawn_frames = 0

    def _load_cnn(self):
        try:
            from tensorflow.keras.models import load_model
            if config.CNN_MODEL_PATH.exists():
                self.cnn_model = load_model(str(config.CNN_MODEL_PATH))
                self.cnn_available = True
                logger.info("CNN model loaded successfully.")
            else:
                logger.warning(f"CNN model file not found: {config.CNN_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")

    @staticmethod
    def _eye_aspect_ratio(eye_points, landmarks, w, h):
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        A = distance.euclidean(pt(eye_points[1]), pt(eye_points[5]))
        B = distance.euclidean(pt(eye_points[2]), pt(eye_points[4]))
        C = distance.euclidean(pt(eye_points[0]), pt(eye_points[3]))
        return (A + B) / (2.0 * C)

    @staticmethod
    def _mouth_aspect_ratio(mouth_points, landmarks, w, h):
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        A = distance.euclidean(pt(mouth_points[1]), pt(mouth_points[7]))
        B = distance.euclidean(pt(mouth_points[2]), pt(mouth_points[6]))
        C = distance.euclidean(pt(mouth_points[3]), pt(mouth_points[5]))
        D = distance.euclidean(pt(mouth_points[0]), pt(mouth_points[4]))
        return (A + B + C) / (2.0 * D)

    @staticmethod
    def _extract_eye_region(frame, eye_points, landmarks, w, h, padding=10):
        def pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        points = np.array([pt(idx) for idx in eye_points])
        
        x_min = int(max(0, np.min(points[:, 0]) - padding))
        x_max = int(min(w, np.max(points[:, 0]) + padding))
        y_min = int(max(0, np.min(points[:, 1]) - padding))
        y_max = int(min(h, np.max(points[:, 1]) + padding))
        
        if x_min >= x_max or y_min >= y_max:
            return None
            
        return frame[y_min:y_max, x_min:x_max]

    @staticmethod
    def _preprocess_eye_for_cnn(eye_region):
        if eye_region is None or eye_region.size == 0:
            return None
        try:
            if len(eye_region.shape) == 3:
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            else:
                eye_gray = eye_region
                
            eye_resized = cv2.resize(eye_gray, (config.CNN_INPUT_WIDTH, config.CNN_INPUT_HEIGHT))
            eye_normalized = eye_resized.astype('float32') / 255.0
            return eye_normalized.reshape(1, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH, config.CNN_INPUT_CHANNELS)
        except Exception as e:
            logger.error(f"Error preprocessing eye region: {e}")
            return None

    def start_calibration(self):
        self.is_calibrating = True
        self.is_calibrated = False
        self.calibration_samples = []

    def update_calibration(self, ear):
        if self.is_calibrating:
            self.calibration_samples.append(ear)
            if len(self.calibration_samples) >= config.CALIBRATION_FRAMES:
                self.finish_calibration()

    def finish_calibration(self):
        if self.calibration_samples:
            self.baseline_ear = float(np.mean(self.calibration_samples))
            self.calibrated_threshold = self.baseline_ear * (config.CALIBRATION_EAR_THRESHOLD_PERCENT / 100.0)
            self.is_calibrated = True
        self.is_calibrating = False
        self.calibration_samples = []

    def reset_calibration(self):
        self.is_calibrated = False
        self.is_calibrating = False
        self.calibration_samples = []
        self.baseline_ear = config.DEFAULT_BASELINE_EAR
        self.calibrated_threshold = config.STATIC_EAR_THRESHOLD
        
    def process_frame(self, frame):
        result = {
            "face_detected": False,
            "landmarks": None,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "ear": 0.0,
            "mar": 0.0,
            "left_cnn_confidence": 0.0,
            "right_cnn_confidence": 0.0,
            "cnn_available": self.cnn_available,
            "cnn_eye_state": "CNN_UNAVAILABLE" if not self.cnn_available else "UNKNOWN",
            "eyes_closed": False,
            "closed_frames": self.closed_frames,
            "yawning": False,
            "yawn_frames": self.yawn_frames,
            "baseline_ear": self.baseline_ear,
            "calibrated_threshold": self.calibrated_threshold,
            "is_calibrated": self.is_calibrated,
            "state": "NORMAL"
        }

        if frame is None:
            return result

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            mp_results = self.face_mesh.process(rgb)
        except Exception as e:
            logger.error(f"MediaPipe processing error: {e}")
            return result

        if mp_results.multi_face_landmarks:
            result["face_detected"] = True
            landmarks = mp_results.multi_face_landmarks[0].landmark
            result["landmarks"] = landmarks

            # Geometric calculations
            left_ear = self._eye_aspect_ratio(self.LEFT_EYE, landmarks, w, h)
            right_ear = self._eye_aspect_ratio(self.RIGHT_EYE, landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = self._mouth_aspect_ratio(self.MOUTH, landmarks, w, h)

            result["left_ear"] = left_ear
            result["right_ear"] = right_ear
            result["ear"] = avg_ear
            result["mar"] = mar

            if self.is_calibrating:
                self.update_calibration(avg_ear)
                result["state"] = "CALIBRATING"
                return result

            # CNN calculations
            cnn_alert = False
            if self.cnn_available and self.cnn_model is not None:
                try:
                    left_eye_region = self._extract_eye_region(frame, self.LEFT_EYE, landmarks, w, h)
                    right_eye_region = self._extract_eye_region(frame, self.RIGHT_EYE, landmarks, w, h)
                    
                    left_eye_input = self._preprocess_eye_for_cnn(left_eye_region)
                    right_eye_input = self._preprocess_eye_for_cnn(right_eye_region)
                    
                    if left_eye_input is not None and right_eye_input is not None:
                        left_pred = self.cnn_model.predict(left_eye_input, verbose=0)[0]
                        right_pred = self.cnn_model.predict(right_eye_input, verbose=0)[0]
                        
                        # Confidence for "closed" class (index 1)
                        result["left_cnn_confidence"] = float(left_pred[1] * 100)
                        result["right_cnn_confidence"] = float(right_pred[1] * 100)
                        
                        if result["left_cnn_confidence"] > config.CNN_CONFIDENCE_THRESHOLD and \
                           result["right_cnn_confidence"] > config.CNN_CONFIDENCE_THRESHOLD:
                            cnn_alert = True
                            result["cnn_eye_state"] = "CLOSED"
                        else:
                            result["cnn_eye_state"] = "OPEN"
                except Exception as e:
                    logger.error(f"CNN prediction error: {e}")

            # Eye closure logic
            threshold = self.calibrated_threshold if self.is_calibrated else config.STATIC_EAR_THRESHOLD
            geometric_alert = (avg_ear < threshold)
            
            # Hybrid alert triggering (AND strategy)
            # Drowsy candidate if:
            # 1. EAR says closed AND CNN is unavailable (fallback)
            # OR 2. EAR says closed AND CNN confirms closed
            drowsy_candidate = False
            if geometric_alert:
                if not self.cnn_available or self.cnn_model is None:
                    drowsy_candidate = True  # EAR-only fallback
                elif cnn_alert:
                    drowsy_candidate = True  # Both EAR and CNN say closed
                else:
                    drowsy_candidate = False # CNN strongly says open, suppressing EAR

            if drowsy_candidate:
                self.closed_frames += 1
                if self.closed_frames >= config.EAR_CONSECUTIVE_FRAMES:
                    result["eyes_closed"] = True
                    result["state"] = "DROWSY"
            else:
                self.closed_frames = 0
            
            result["closed_frames"] = self.closed_frames

            # Yawn logic
            if mar > config.MAR_THRESHOLD:
                self.yawn_frames += 1
                if self.yawn_frames >= config.MAR_CONSECUTIVE_FRAMES:
                    result["yawning"] = True
                    if result["state"] == "NORMAL":
                        result["state"] = "YAWNING"
            else:
                self.yawn_frames = 0
                
            result["yawn_frames"] = self.yawn_frames

        return result

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()
        self.cnn_model = None
