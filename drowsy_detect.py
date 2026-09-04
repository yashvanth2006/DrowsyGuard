import cv2
import time
import pygame
import numpy as np
import sys
from core.detector import DrowsyDetector

# ── Audio setup ───────────────────────────────────────────
alert_sound = None
try:
    pygame.mixer.init()
    sample_rate = 44100
    duration = 0.5
    frequency = 1000
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    alert_sound = pygame.sndarray.make_sound(stereo)
except Exception as e:
    print(f"[WARNING] Audio initialization failed: {e}")

def play_alert():
    if alert_sound:
        alert_sound.play()

last_alert = 0

def main():
    global last_alert
    
    # Initialize the detector
    detector = DrowsyDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        print("Please check that your webcam is connected and not being used by another application.")
        detector.close()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        sys.exit(1)
        
    print("[INFO] DrowsyGuard v2 started - press Q to quit, C to calibrate, R to reset calibration")

    consecutive_failures = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print("[ERROR] Camera disconnected or failed to read 5 times consecutively. Exiting.")
                    break
                time.sleep(0.1) # slight delay before retry
                continue
            
            # Reset on success
            consecutive_failures = 0

            try:
                result = detector.process_frame(frame)
            except Exception as e:
                print(f"[ERROR] Detector processing failed: {type(e).__name__}: {e}")
                break
            
            status_text = "NORMAL"
            status_color = (0, 255, 0)
            alert = False

            if result.get("face_detected"):
                # ── Display values ──
                avg_ear = result.get("ear", 0.0)
                mar = result.get("mar", 0.0)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                y_offset = 100
                
                # CNN Status
                if result.get("cnn_available"):
                    cnn_state = result.get("cnn_eye_state", "UNKNOWN")
                    left_conf = result.get("left_cnn_confidence", 0.0)
                    right_conf = result.get("right_cnn_confidence", 0.0)
                    cv2.putText(frame, f"CNN: {cnn_state} (L:{left_conf:.0f}% R:{right_conf:.0f}%)", (30, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 30

                state = result.get("state", "NORMAL")
                if state == "CALIBRATING":
                    status_text = "CALIBRATING..."
                    status_color = (255, 255, 0)
                elif state == "DROWSY":
                    status_text = "DROWSY! EYES CLOSED!"
                    status_color = (0, 0, 255)
                    alert = True
                elif state == "YAWNING":
                    status_text = "DROWSY! YAWNING!"
                    status_color = (0, 165, 255)
                    alert = True
                else:
                    status_text = state

                cv2.putText(frame, f"Status: {status_text}", (30, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                y_offset += 30

                # ── Alert display ──
                if alert:
                    cv2.putText(frame, "⚠ WAKE UP!", (30, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    # Play beep every 2 seconds
                    if time.time() - last_alert > 2:
                        play_alert()
                        last_alert = time.time()
                
                if result.get("is_calibrated"):
                    cv2.putText(frame, "Calibrated", (frame.shape[1] - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "No face detected", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("DrowsyGuard", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                detector.start_calibration()
            elif key == ord('r'):
                detector.reset_calibration()

    finally:
        cap.release()
        try:
            detector.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            pygame.mixer.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()