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

if "app_state_dict" not in st.session_state:
    st.session_state.app_state_dict = {
        "monitoring": False,
        "risk_level": "Unknown",
        "drowsiness": False,
        "alert_count": 0,
        "session_duration": "0 minutes"
    }

if "command_queue" not in st.session_state:
    import queue
    st.session_state.command_queue = queue.Queue()

if "nova_status" not in st.session_state:
    st.session_state.nova_status = "○ Disabled"

state = st.session_state.state
app_state_dict = st.session_state.app_state_dict

from core.detector import DrowsyDetector
from voice_assistant import VoiceAssistant
import config

if "voice_assistant" not in st.session_state:
    def state_getter():
        return st.session_state.app_state_dict
    st.session_state.voice_assistant = VoiceAssistant(st.session_state.command_queue, state_getter)
    st.session_state.voice_assistant.start()

# ── Audio Setup ────────────────────────────────────────────
pygame.mixer.init()

def play_alert_sound(freq=1000, duration=0.3):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
    sound.play()

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
    
    st.markdown("#### 🎙️ Voice Assistant")
    st.markdown(f"**Nova:** {st.session_state.nova_status}")
    
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
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>Eye Openness</div><div class='metric-value'>0%</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value'>--</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>Alerts</div><div class='metric-value'>0</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>CNN Status</div><div class='metric-value'>WAIT</div></div>", unsafe_allow_html=True)
    
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
    if state["session_start"]:
        duration = int((time.time() - state["session_start"]) / 60)
        state["session_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration": duration,
            "alerts": state["alert_count"]
        })

if calibrate_btn:
    state["calibrated"] = False

# Process pending voice commands (Non-blocking)
from queue import Empty
try:
    while True:
        cmd = st.session_state.command_queue.get_nowait()
        action = cmd.get("action")
        if action == "START":
            state["detection_active"] = True
            if not state["session_start"]:
                state["session_start"] = time.time()
                state["alert_count"] = 0
            st.rerun()
        elif action == "STOP":
            state["detection_active"] = False
            st.rerun()
        elif action == "NOVA_STATUS":
            st.session_state.nova_status = cmd.get("status", "○ Disabled")
            st.rerun()
except Empty:
    pass

# ── Detection Loop ─────────────────────────────────────────
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
        
        # Check queue during camera loop to stay responsive
        try:
            while True:
                cmd = st.session_state.command_queue.get_nowait()
                action = cmd.get("action")
                if action == "STOP":
                    state["detection_active"] = False
                elif action == "NOVA_STATUS":
                    st.session_state.nova_status = cmd.get("status", "○ Disabled")
        except Empty:
            pass
        
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
                
            # Sync state dict for VoiceAssistant
            dur = "unknown time"
            if state["session_start"]:
                elapsed = int(time.time() - state["session_start"])
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                dur = f"{hours} hour{'s' if hours>1 else ''} and {minutes} minutes" if hours > 0 else f"{minutes} minute{'s' if minutes!=1 else ''}"
                    
            st.session_state.app_state_dict.update({
                "monitoring": state["detection_active"],
                "risk_level": risk_level,
                "drowsiness": result.get("eyes_closed", False),
                "alert_count": state["alert_count"],
                "session_duration": dur
            })
    
    detector.close()
    if cap:
        cap.release()
else:
    # Idle loop checking for voice commands when camera is off
    import time
    time.sleep(0.5)
    st.rerun()
