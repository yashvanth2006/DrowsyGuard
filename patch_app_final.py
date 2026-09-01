import re

with open("app_final.py", "r", encoding="utf-8") as f:
    content = f.read()

# Chunk 1: Remove redundant CNN and MediaPipe logic
chunk1_regex = re.compile(r"# ── CNN Model Loading \(Optional\) ───────────────────────────.*?return \(A \+ B\) / \(2\.0 \* C\)", re.DOTALL)
chunk1_replacement = """from core.detector import DrowsyDetector
import config

# ── Audio Setup ────────────────────────────────────────────
pygame.mixer.init()

def play_alert_sound(freq=1000, duration=0.3):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
    sound.play()"""

content = chunk1_regex.sub(chunk1_replacement, content)

# Chunk 2: Metrics container
chunk2_regex = re.compile(r"# Metrics.*?m3\.markdown\(f\"<div class='metric-card'><div class='metric-label'>Alerts</div><div class='metric-value'>0</div></div>\", unsafe_allow_html=True\)", re.DOTALL)
chunk2_replacement = """# Metrics
    metrics_container = st.empty()
    with metrics_container.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Eye Openness</div><div class='metric-value'>0%</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value'>--</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Alerts</div><div class='metric-value'>0</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>CNN Status</div><div class='metric-value'>WAIT</div></div>", unsafe_allow_html=True)"""

content = chunk2_regex.sub(chunk2_replacement, content)

# Chunk 3: The Detection Loop
chunk3_regex = re.compile(r"# ── Detection Loop ─────────────────────────────────────────.*", re.DOTALL)
chunk3_replacement = """# ── Detection Loop ─────────────────────────────────────────
if state["detection_active"]:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = DrowsyDetector()
    if state["calibrated"]:
        detector.baseline_ear = state["baseline_ear"]
        detector.calibrated_threshold = state["baseline_ear"] * (ear_threshold / 100.0)
        detector.is_calibrated = True
    
    last_alert = 0
    frame_count = 0
    
    # Calibration mode
    if not state["calibrated"]:
        st.info("🎯 CALIBRATION: Keep eyes open for 5 seconds...")
        detector.start_calibration()
    
    while cap.isOpened() and state["detection_active"]:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sync dynamic settings to detector
        config.STATIC_EAR_THRESHOLD = state.get("baseline_ear", 0.3) * (ear_threshold / 100.0)
        config.EAR_CONSECUTIVE_FRAMES = frame_threshold
        if state["calibrated"]:
            detector.calibrated_threshold = config.STATIC_EAR_THRESHOLD

        result = detector.process_frame(frame)
        
        if result["state"] == "CALIBRATING":
            if detector.is_calibrated:
                state["baseline_ear"] = detector.baseline_ear
                state["calibrated"] = True
                st.success(f"✅ Calibration complete! Baseline EAR: {state['baseline_ear']:.3f}")
        else:
            state["current_ear"] = result["ear"]
            
            # Alert Sound Logic
            if result["eyes_closed"] or result["yawning"]:
                if time.time() - last_alert > 2:
                    state["alert_count"] += 1
                    play_alert_sound()
                    last_alert = time.time()

        # Draw overlay
        h, w = frame.shape[:2]
        if result["face_detected"] and result.get("landmarks"):
            for idx in detector.LEFT_EYE + detector.RIGHT_EYE:
                lm = result["landmarks"][idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 136), -1)
            
            color = (255, 51, 68) if result["eyes_closed"] else (0, 255, 136)
            cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 2)
        else:
            cv2.putText(frame, "No Face", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display Video
        frame_resized = cv2.resize(frame, (640, 480))
        ret_jpg, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret_jpg:
            video_container.image(buffer.tobytes(), channels="BGR", width='stretch')
        
        # Update UI Metrics (throttled)
        if frame_count % 5 == 0 and state["calibrated"]:
            openness = int((state["current_ear"] / state["baseline_ear"]) * 100) if state["baseline_ear"] > 0 else 0
            risk_level, risk_color = get_risk_level(state["current_ear"], state["baseline_ear"])
            
            cnn_status = "● Active" if result["cnn_available"] else "○ N/A"
            cnn_color = "#00ff88" if result["cnn_available"] else "#888"
            
            with metrics_container.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f"<div class='metric-card'><div class='metric-label'>Eye Openness</div><div class='metric-value'>{openness}%</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value' style='color:{risk_color}'>{risk_level}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-label'>Alerts</div><div class='metric-value'>{state['alert_count']}</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='metric-label'>CNN Status</div><div class='metric-value' style='color:{cnn_color}; font-size:18px;'>{cnn_status}</div></div>", unsafe_allow_html=True)
            
            if result["eyes_closed"]:
                alert_container.markdown("<div class='alert-danger'>🚨 WAKE UP! Eyes detected closed!</div>", unsafe_allow_html=True)
            elif openness < 80:
                alert_container.markdown("<div class='alert-warning'>⚠️ Fatigue detected. Stay alert.</div>", unsafe_allow_html=True)
            else:
                alert_container.markdown("<div class='alert-safe'>✅ Driver alert and focused.</div>", unsafe_allow_html=True)
    
    detector.close()
    if cap:
        cap.release()
"""

content = chunk3_regex.sub(chunk3_replacement, content)

with open("app_final.py", "w", encoding="utf-8") as f:
    f.write(content)
