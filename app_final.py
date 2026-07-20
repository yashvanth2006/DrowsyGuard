import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import pygame
import time
import streamlit as st
import threading
import os
import json
from datetime import datetime

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; }

.metric-card {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}
.metric-value { font-size: 28px; font-weight: 700; color: #00d4ff; }
.metric-label { font-size: 11px; color: #888; text-transform: uppercase; }

.alert-safe { background: rgba(0,255,136,0.1); border-left: 4px solid #00ff88; padding: 15px; border-radius: 8px; }
.alert-warning { background: rgba(255,136,0,0.1); border-left: 4px solid #ff8800; padding: 15px; border-radius: 8px; }
.alert-danger { background: rgba(255,51,68,0.15); border-left: 4px solid #ff3344; padding: 15px; border-radius: 8px; animation: blink 0.5s infinite; }
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.7; } }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0066aa);
    color: white; border: none; border-radius: 8px; font-weight: 600; padding: 10px 20px;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,212,255,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "detection_active": False,
        "paused": False,
        "calibrated": False,
        "baseline_ear": 0.30,
        "current_ear": 0.30,
        "session_start": None,
        "alert_count": 0,
        "emergency_contact": "",
        "session_history": []
    }

state = st.session_state.state

# ── CNN Model Loading (Optional) ───────────────────────────
@st.cache_resource
def load_cnn_model():
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists("eye_state_model.h5"):
            return load_model("eye_state_model.h5")
    except:
        pass
    return None

cnn_model = load_cnn_model()

# ── Audio Setup ────────────────────────────────────────────
pygame.mixer.init()

def play_alert_sound(freq=1000, duration=0.3):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
    sound.play()

# ── MediaPipe Setup ───────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye_points, landmarks, w, h):
    def pt(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
    A = distance.euclidean(pt(eye_points[1]), pt(eye_points[5]))
    B = distance.euclidean(pt(eye_points[2]), pt(eye_points[4]))
    C = distance.euclidean(pt(eye_points[0]), pt(eye_points[3]))
    return (A + B) / (2.0 * C)

def get_risk_level(ear, baseline):
    ratio = ear / baseline if baseline > 0 else 1
    if ratio > 0.85: return "LOW", "#00ff88"
    elif ratio > 0.70: return "MEDIUM", "#ff8800"
    else: return "HIGH", "#ff3344"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Calibration
    if not state["calibrated"]:
        st.markdown("#### 🔧 Calibration Required")
        st.info("Click 'Calibrate' while looking at the camera with eyes fully open.")
        if st.button("🎯 Calibrate Baseline"):
            state["calibrated"] = True
            state["baseline_ear"] = 0.30  # Will be updated during calibration
            st.success("Calibration mode active. Keep eyes open for 5 seconds.")
    
    # Thresholds
    ear_threshold = st.slider("Alert Threshold (% of baseline)", 50, 90, 70)
    frame_threshold = st.slider("Alert Sensitivity (frames)", 5, 30, 15)
    
    # Emergency Contact
    st.markdown("#### 🆘 Emergency")
    emergency_contact = st.text_input("Emergency Contact", value=state["emergency_contact"])
    state["emergency_contact"] = emergency_contact
    
    # Session History
    st.markdown("#### 📊 Recent Sessions")
    if state["session_history"]:
        for session in state["session_history"][-5:]:
            st.markdown(f"- {session['date']}: {session['duration']} min, {session['alerts']} alerts")

# ── Main Layout ───────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    video_container = st.empty()
    
    # Metrics
    metrics_container = st.empty()
    with metrics_container.container():
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Eye Openness</div><div class='metric-value'>0%</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value'>--</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Alerts</div><div class='metric-value'>0</div></div>", unsafe_allow_html=True)
    
    alert_container = st.empty()
    alert_container.markdown("<div class='alert-safe'>✅ Ready to start. Click 'Start Monitoring' below.</div>", unsafe_allow_html=True)
    
    # Controls
    c1, c2, c3 = st.columns(3)
    start_btn = c1.button("▶ Start Monitoring", key="main_start")
    stop_btn = c2.button("⏹ Stop", key="main_stop")
    calibrate_btn = c3.button("🎯 Recalibrate", key="main_calibrate")

with col2:
    st.markdown("### 📋 Quick Guide")
    st.markdown("""
    **How it works:**
    1. Click 'Start Monitoring'
    2. Keep camera facing you
    3. System alerts when eyes close
    
    **Alert Threshold:** {}% of your baseline
    **Sensitivity:** {} frames
    
    **Tips:**
    - Good lighting improves accuracy
    - Keep camera at eye level
    - Recalibrate if glasses are added/removed
    """.format(ear_threshold, frame_threshold))
    
    st.markdown("### 🆘 Emergency")
    if state["emergency_contact"]:
        st.success(f"Contact: {state['emergency_contact']}")
        if st.button("📞 Call Emergency"):
            st.warning(f"Would call: {state['emergency_contact']}")
    else:
        st.warning("No emergency contact set")

# ── Control Logic ─────────────────────────────────────────
if start_btn:
    state["detection_active"] = True
    state["session_start"] = time.time()
    state["alert_count"] = 0

if stop_btn:
    state["detection_active"] = False
    # Save session
    if state["session_start"]:
        duration = int((time.time() - state["session_start"]) / 60)
        state["session_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration": duration,
            "alerts": state["alert_count"]
        })

if calibrate_btn:
    state["calibrated"] = False

# ── Detection Loop ─────────────────────────────────────────
if state["detection_active"]:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    ear_values = []
    alert_counter = 0
    last_alert = 0
    frame_count = 0
    calibration_frames = 0
    
    # Calibration mode
    if not state["calibrated"]:
        st.info("🎯 CALIBRATION: Keep eyes open for 5 seconds...")
    
    while cap.isOpened() and state["detection_active"]:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(LEFT_EYE, lm, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, lm, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            state["current_ear"] = avg_ear
            
            # Calibration
            if not state["calibrated"]:
                ear_values.append(avg_ear)
                calibration_frames += 1
                
                if calibration_frames >= 150:  # 5 seconds at 30fps
                    state["baseline_ear"] = np.mean(ear_values)
                    state["calibrated"] = True
                    st.success(f"✅ Calibration complete! Baseline EAR: {state['baseline_ear']:.3f}")
                    ear_values = []
            else:
                # Detection
                threshold = state["baseline_ear"] * (ear_threshold / 100.0)
                
                if avg_ear < threshold:
                    alert_counter += 1
                    if alert_counter >= frame_threshold:
                        if time.time() - last_alert > 2:
                            state["alert_count"] += 1
                            play_alert_sound()
                            last_alert = time.time()
                else:
                    alert_counter = 0
                
                # Draw
                for idx in LEFT_EYE + RIGHT_EYE:
                    x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 136), -1)
                
                color = (255, 51, 68) if alert_counter >= frame_threshold else (0, 255, 136)
                cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 2)
        else:
            cv2.putText(frame, "No Face", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Update video
        frame_resized = cv2.resize(frame, (640, 480))
        ret_jpg, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret_jpg:
            video_container.image(buffer.tobytes(), channels="BGR", width='stretch')
        
        # Update UI every 5 frames
        if frame_count % 5 == 0 and state["calibrated"]:
            openness = int((state["current_ear"] / state["baseline_ear"]) * 100) if state["baseline_ear"] > 0 else 0
            risk_level, risk_color = get_risk_level(state["current_ear"], state["baseline_ear"])
            
            with metrics_container.container():
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-card'><div class='metric-label'>Eye Openness</div><div class='metric-value'>{openness}%</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value' style='color:{risk_color}'>{risk_level}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-label'>Alerts</div><div class='metric-value'>{state['alert_count']}</div></div>", unsafe_allow_html=True)
            
            if alert_counter >= frame_threshold:
                alert_container.markdown("<div class='alert-danger'>🚨 WAKE UP! Eyes detected closed!</div>", unsafe_allow_html=True)
            elif openness < 80:
                alert_container.markdown("<div class='alert-warning'>⚠️ Fatigue detected. Stay alert.</div>", unsafe_allow_html=True)
            else:
                alert_container.markdown("<div class='alert-safe'>✅ Driver alert and focused.</div>", unsafe_allow_html=True)
    
    if cap:
        cap.release()
