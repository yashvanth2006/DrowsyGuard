import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import pygame
import time

# ── Audio setup ───────────────────────────────────────────
pygame.mixer.init()

def play_alert():
    # Generates a beep sound without needing an audio file
    import numpy as np
    sample_rate = 44100
    duration    = 0.5
    frequency   = 1000
    t      = np.linspace(0, duration, int(sample_rate * duration))
    wave   = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    sound  = pygame.sndarray.make_sound(stereo)
    sound.play()

# ── EAR formula ──────────────────────────────────────────
def eye_aspect_ratio(eye_points, landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    A = distance.euclidean(pt(eye_points[1]), pt(eye_points[5]))
    B = distance.euclidean(pt(eye_points[2]), pt(eye_points[4]))
    C = distance.euclidean(pt(eye_points[0]), pt(eye_points[3]))
    return (A + B) / (2.0 * C)

# ── MAR formula ──────────────────────────────────────────
def mouth_aspect_ratio(mouth_points, landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    A = distance.euclidean(pt(mouth_points[1]), pt(mouth_points[7]))
    B = distance.euclidean(pt(mouth_points[2]), pt(mouth_points[6]))
    C = distance.euclidean(pt(mouth_points[3]), pt(mouth_points[5]))
    D = distance.euclidean(pt(mouth_points[0]), pt(mouth_points[4]))
    return (A + B + C) / (2.0 * D)

# ── MediaPipe setup ───────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 39, 269, 405, 291, 375, 321, 308]

# ── Thresholds ────────────────────────────────────────────
EAR_THRESHOLD    = 0.25
MAR_THRESHOLD    = 0.6
FRAME_THRESHOLD  = 20
YAWN_THRESHOLD   = 15

ear_counter  = 0
yawn_counter = 0
last_alert   = 0

# ── Webcam loop ───────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("✅ DrowsyGuard v2 started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text  = "ALERT"
    status_color = (0, 255, 0)
    alert        = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # ── EAR ──
        left_ear  = eye_aspect_ratio(LEFT_EYE,  landmarks, w, h)
        right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        # ── MAR ──
        mar = mouth_aspect_ratio(MOUTH, landmarks, w, h)

        # ── Display values ──
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Drowsiness logic ──
        if avg_ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= FRAME_THRESHOLD:
                status_text  = "DROWSY! EYES CLOSED!"
                status_color = (0, 0, 255)
                alert        = True
        else:
            ear_counter = 0

        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= YAWN_THRESHOLD:
                status_text  = "DROWSY! YAWNING!"
                status_color = (0, 165, 255)
                alert        = True
        else:
            yawn_counter = 0

        # ── Alert display ──
        if alert:
            cv2.putText(frame, "⚠ WAKE UP!", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Play beep every 2 seconds
            if time.time() - last_alert > 2:
                play_alert()
                last_alert = time.time()

        cv2.putText(frame, f"Status: {status_text}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    else:
        cv2.putText(frame, "No face detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("DrowsyGuard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()