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
from queue import Empty

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard Pro",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg-primary: #070A12;
    --bg-secondary: #0A0E18;
    --surface: #0D111C;
    --border: rgba(0, 212, 255, 0.15);
    --text-primary: #FFFFFF;
    --text-secondary: #8B949E;
    --accent: #00D4FF;
    --success: #00FF88;
    --warning: #FFB300;
    --danger: #FF3344;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', 'SF Pro Display', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem; max-width: 1440px; }

/* Hide Streamlit default elements */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* Typography */
.text-muted { color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
.text-mono { font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace; }
.text-accent { color: var(--accent); }

/* Header */
.dg-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.dg-title { font-size: 1.8rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.2; letter-spacing: -0.02em; }
.dg-subtitle { font-size: 0.8rem; color: var(--text-secondary); letter-spacing: 0.2em; }
.dg-sys-status { font-size: 0.8rem; font-weight: 600; display: flex; flex-direction: column; text-align: right; }

/* Panels & Cards */
.dg-panel {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.dg-panel-title {
    font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 1rem;
}

/* Camera */
.camera-wrapper {
    position: relative;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: #000;
    margin-bottom: 1rem;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.05);
}
.camera-overlay-top-left { position: absolute; top: 10px; left: 15px; color: var(--success); font-weight: 600; font-size: 0.8rem; text-shadow: 0 1px 4px rgba(0,0,0,0.8); z-index: 10; }
.camera-overlay-top-right { position: absolute; top: 10px; right: 15px; color: rgba(255,255,255,0.8); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-shadow: 0 1px 4px rgba(0,0,0,0.8); z-index: 10; }
.camera-overlay-bottom-left { position: absolute; bottom: 10px; left: 15px; color: rgba(255,255,255,0.8); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-shadow: 0 1px 4px rgba(0,0,0,0.8); z-index: 10; }
.camera-overlay-bottom-right { position: absolute; bottom: 10px; right: 15px; color: var(--accent); font-size: 0.8rem; font-family: monospace; font-weight: 600; text-shadow: 0 1px 4px rgba(0,0,0,0.8); z-index: 10; }

/* Metrics */
.dg-metric {
    background-color: rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.dg-metric-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.dg-metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text-primary); line-height: 1; }

/* Safety States */
.safety-panel { padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; }
.safety-safe { background: rgba(0,255,136,0.05); border-left: 4px solid var(--success); }
.safety-warning { background: rgba(255,179,0,0.05); border-left: 4px solid var(--warning); }
.safety-danger { background: rgba(255,51,68,0.1); border-left: 4px solid var(--danger); animation: pulse-border 1.5s infinite; }

@keyframes pulse-border {
    0% { box-shadow: 0 0 0 0 rgba(255,51,68,0.2); }
    70% { box-shadow: 0 0 0 10px rgba(255,51,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,51,68,0); }
}

.safety-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.25rem; }
.safe-text { color: var(--success); }
.warning-text { color: var(--warning); }
.danger-text { color: var(--danger); }

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.btn-primary > button { background: linear-gradient(135deg, var(--accent), #0066aa); color: white; border: none; }
.btn-primary > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,212,255,0.3); color: white; }
.btn-danger > button { background: rgba(255,51,68,0.1); color: var(--danger); border: 1px solid rgba(255,51,68,0.3); }
.btn-danger > button:hover { background: rgba(255,51,68,0.2); color: var(--danger); border: 1px solid var(--danger); }
.btn-secondary > button { background: rgba(255,255,255,0.05); color: var(--text-primary); border: 1px solid rgba(255,255,255,0.1); }
.btn-secondary > button:hover { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; }

/* Status List */
.status-row { display: flex; justify-content: space-between; margin-bottom: 0.5rem; align-items: center; font-size: 0.85rem; }
.status-label { color: var(--text-secondary); }
.status-val { font-weight: 600; }

/* Progress/Risk bar */
.risk-bar-bg { background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 5px; }
.risk-bar-fill { height: 100%; transition: width 0.3s ease; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "detection_active": False,
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
        "session_duration": "00:00:00"
    }

if "command_queue" not in st.session_state:
    import queue
    st.session_state.command_queue = queue.Queue()

if "nova_status" not in st.session_state:
    st.session_state.nova_status = "○ OFFLINE"

state = st.session_state.state
app_state_dict = st.session_state.app_state_dict

from core.detector import DrowsyDetector
from voice_assistant import VoiceAssistant
from analytics.logger import SessionLogger
import config

if "voice_assistant" not in st.session_state:
    def state_getter():
        return st.session_state.app_state_dict
    st.session_state.voice_assistant = VoiceAssistant(st.session_state.command_queue, state_getter)
    st.session_state.voice_assistant.start()

if "session_logger" not in st.session_state:
    st.session_state.session_logger = SessionLogger()

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
    if ratio > 0.85: return "LOW", "var(--success)"
    elif ratio > 0.70: return "MEDIUM", "var(--warning)"
    else: return "HIGH", "var(--danger)"

def format_duration(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='dg-panel-title'>MONITORING</div>", unsafe_allow_html=True)
    ear_threshold = st.slider("Alert Threshold (% of baseline)", 50, 90, 70)
    frame_threshold = st.slider("Alert Sensitivity (frames)", 5, 30, 15)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
    st.markdown("<div class='dg-panel-title'>SYSTEM SETTINGS</div>", unsafe_allow_html=True)
    emergency_contact = st.text_input("Emergency Contact", value=state["emergency_contact"], placeholder="+1 234 567 8900")
    state["emergency_contact"] = emergency_contact
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1)'>", unsafe_allow_html=True)
    st.markdown("<div class='dg-panel-title'>NOVA ASSISTANT</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status-label'>Status: <span style='color:var(--accent); font-weight:600;'>{st.session_state.nova_status}</span></div>", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────
dur_str = "00:00:00"
if state["session_start"] and state["detection_active"]:
    dur_str = format_duration(int(time.time() - state["session_start"]))

sys_status = "● SYSTEM ONLINE" if state["detection_active"] else ("● CALIBRATION REQUIRED" if not state["calibrated"] else "○ SYSTEM STANDBY")
sys_color = "var(--success)" if state["detection_active"] else ("var(--warning)" if not state["calibrated"] else "var(--text-secondary)")

st.markdown(f"""
<div class='dg-header'>
    <div>
        <div class='dg-title'>DROWSYGUARD <span style='color:var(--accent); font-weight:300;'>PRO</span></div>
        <div class='dg-subtitle'>AI DRIVER SAFETY SYSTEM</div>
    </div>
    <div class='dg-sys-status'>
        <div style='color:{sys_color}; font-size:0.85rem; letter-spacing:0.05em;'>{sys_status}</div>
        <div style='color:var(--text-secondary); font-family:monospace; margin-top:2px;'>SESSION {dur_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Process pending voice commands (Non-blocking) ──
try:
    while True:
        cmd = st.session_state.command_queue.get_nowait()
        action = cmd.get("action")
        st.session_state.session_logger.log_event("VOICE_COMMAND", {"action": action})
        if action == "START":
            state["detection_active"] = True
            if not state["session_start"]:
                state["session_start"] = time.time()
                state["alert_count"] = 0
                st.session_state.session_logger.start_session()
            st.rerun()
        elif action == "STOP":
            state["detection_active"] = False
            st.session_state.session_logger.end_session()
            st.rerun()
        elif action == "NOVA_STATUS":
            st.session_state.nova_status = cmd.get("status", "○ OFFLINE").upper()
            st.rerun()
except Empty:
    pass

# ── Layout Containers ───────────────────────────────────────────
main_col, side_col = st.columns([2.2, 1])

with main_col:
    # We will inject the video feed here during the loop, so create empty placeholder
    camera_placeholder = st.empty()
    
    metrics_placeholder = st.empty()
    safety_state_placeholder = st.empty()
    
    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
        start_btn = st.button("START MONITORING" if state["calibrated"] else "START (UNCALIBRATED)", key="btn_start")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='btn-secondary'>", unsafe_allow_html=True)
        calibrate_btn = st.button("RECALIBRATE", key="btn_calib")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='btn-danger'>", unsafe_allow_html=True)
        stop_btn = st.button("STOP", key="btn_stop")
        st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    ai_status_placeholder = st.empty()
    telemetry_placeholder = st.empty()
    
    # Emergency / Nova / Quick Guide
    st.markdown("""
    <div class='dg-panel'>
        <div class='dg-panel-title'>EMERGENCY CONTACT</div>
        {}
    </div>
    """.format(
        f"<div class='status-val' style='font-size:1.1rem; color:var(--text-primary);'>{state['emergency_contact']}</div><div class='text-muted' style='margin-top:5px;'><span style='color:var(--success)'>● READY</span></div>" 
        if state["emergency_contact"] else 
        "<div class='text-muted'>No contact configured.<br>Update in settings.</div>"
    ), unsafe_allow_html=True)
    
    st.markdown("""
    <div class='dg-panel'>
        <div class='dg-panel-title'>HOW IT WORKS</div>
        <div style='font-size:0.85rem; color:var(--text-secondary); line-height:1.5;'>
            <div style='margin-bottom:8px;'><strong style='color:var(--text-primary)'>01</strong> Calibrate baseline</div>
            <div style='margin-bottom:8px;'><strong style='color:var(--text-primary)'>02</strong> Start monitoring</div>
            <div style='margin-bottom:8px;'><strong style='color:var(--text-primary)'>03</strong> Stay focused. Drowsiness triggers alerts.</div>
            <div style='margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.05); color:var(--accent); font-size:0.8rem;'>TIP: Good lighting improves accuracy.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent Sessions
    history = st.session_state.session_logger.get_session_history()
    if history:
        sess_html = "<div class='dg-panel'><div class='dg-panel-title'>RECENT SESSIONS</div>"
        for sess in history[:3]:
            sess_html += f"""
            <div style='display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:10px;'>
                <div>
                    <div style='color:var(--text-primary); font-weight:600;'>{sess['date']}</div>
                    <div style='color:var(--text-secondary);'>{sess['duration']} min</div>
                </div>
                <div style='text-align:right;'>
                    <div style='color:var(--warning); font-weight:600;'>{sess['alerts']} ALERTS</div>
                    <div style='color:var(--text-secondary); font-size:0.75rem;'>RISK: {sess['risk']}</div>
                </div>
            </div>
            """
        sess_html += "</div>"
        st.markdown(sess_html, unsafe_allow_html=True)


# ── Control Logic ─────────────────────────────────────────
if start_btn:
    state["detection_active"] = True
    if not state["session_start"]:
        state["session_start"] = time.time()
        state["alert_count"] = 0
        st.session_state.session_logger.start_session()
    st.rerun()

if stop_btn:
    state["detection_active"] = False
    if state["session_start"]:
        duration = int(time.time() - state["session_start"])
        state["session_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "duration": duration // 60,
            "alerts": state["alert_count"]
        })
    st.session_state.session_logger.end_session()
    state["session_start"] = None
    st.rerun()

if calibrate_btn:
    state["calibrated"] = False
    state["detection_active"] = True
    if not state["session_start"]:
        state["session_start"] = time.time()
        st.session_state.session_logger.start_session()
    st.rerun()

# Default empty UI state rendering before/during idle
def render_metrics(openness, risk_level, risk_color, alerts, cnn_status, cnn_color):
    metrics_placeholder.markdown(f"""
    <div class='dg-metric-row'>
        <div class='dg-metric'>
            <div class='dg-metric-label'>EYE OPENNESS</div>
            <div class='dg-metric-value'>{openness}%</div>
        </div>
        <div class='dg-metric'>
            <div class='dg-metric-label'>RISK LEVEL</div>
            <div class='dg-metric-value' style='color:{risk_color}'>{risk_level}</div>
        </div>
        <div class='dg-metric'>
            <div class='dg-metric-label'>ALERTS</div>
            <div class='dg-metric-value'>{alerts}</div>
        </div>
        <div class='dg-metric'>
            <div class='dg-metric-label'>CNN</div>
            <div class='dg-metric-value' style='color:{cnn_color}; font-size:1.4rem; padding-top:5px;'>{cnn_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_safety_state(mode="IDLE", msg="System idle. Ready to monitor."):
    if mode == "IDLE":
        cls, tcls, title = "safety-safe", "safe-text", "● SYSTEM IDLE"
    elif mode == "CALIBRATING":
        cls, tcls, title = "safety-warning", "warning-text", "● CALIBRATION REQUIRED"
    elif mode == "ALERT":
        cls, tcls, title = "safety-safe", "safe-text", "● DRIVER ALERT"
    elif mode == "WARNING":
        cls, tcls, title = "safety-warning", "warning-text", "● ATTENTION"
    elif mode == "DROWSY":
        cls, tcls, title = "safety-danger", "danger-text", "● DROWSINESS DETECTED"
    else:
        cls, tcls, title = "safety-safe", "safe-text", "● DRIVER ALERT"

    safety_state_placeholder.markdown(f"""
    <div class='safety-panel {cls}'>
        <div class='safety-title {tcls}'>{title}</div>
        <div style='color:var(--text-primary); font-size:0.95rem; margin-top:5px;'>{msg}</div>
    </div>
    """, unsafe_allow_html=True)

def render_side_panels(driver_state, driver_color, risk_level, risk_pct, risk_bar_color, cnn_state, ear, mar, cnn_conf):
    ai_status_placeholder.markdown(f"""
    <div class='dg-panel'>
        <div class='dg-panel-title'>AI STATUS</div>
        <div class='status-row'>
            <span class='status-label'>DRIVER STATE</span>
            <span class='status-val' style='color:{driver_color}'>● {driver_state}</span>
        </div>
        <div class='status-row'>
            <span class='status-label'>RISK LEVEL</span>
            <span class='status-val'>{risk_level}</span>
        </div>
        <div class='risk-bar-bg'>
            <div class='risk-bar-fill' style='width:{risk_pct}%; background-color:{risk_bar_color};'></div>
        </div>
        <div class='status-row' style='margin-top:15px;'>
            <span class='status-label'>EYE STATE</span>
            <span class='status-val'>{cnn_state}</span>
        </div>
        <div class='status-row'>
            <span class='status-label'>CALIBRATION</span>
            <span class='status-val' style='color:{"var(--success)" if state["calibrated"] else "var(--warning)"}'>
                {"READY" if state["calibrated"] else "REQUIRED"}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    telemetry_placeholder.markdown(f"""
    <div class='dg-panel'>
        <div class='dg-panel-title'>AI TELEMETRY</div>
        <div class='status-row'>
            <span class='status-label text-mono'>EAR</span>
            <span class='status-val text-mono text-accent'>{ear:.3f}</span>
        </div>
        <div class='status-row'>
            <span class='status-label text-mono'>MAR</span>
            <span class='status-val text-mono text-accent'>{mar:.3f}</span>
        </div>
        <div class='status-row'>
            <span class='status-label text-mono'>CNN CONFIDENCE</span>
            <span class='status-val text-mono text-accent'>{cnn_conf:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initial idle rendering
if not state["detection_active"]:
    camera_placeholder.markdown("""
    <div class='camera-wrapper' style='height:480px; display:flex; align-items:center; justify-content:center; flex-direction:column;'>
        <div style='color:var(--text-secondary); font-size:3rem; margin-bottom:1rem;'>📷</div>
        <div style='color:var(--text-secondary); font-weight:600; letter-spacing:0.1em;'>CAMERA OFFLINE</div>
    </div>
    """, unsafe_allow_html=True)
    render_metrics("--", "--", "var(--text-secondary)", state["alert_count"], "WAIT", "var(--text-secondary)")
    render_safety_state("IDLE", "System idle. Ready to monitor.")
    render_side_panels("OFFLINE", "var(--text-secondary)", "UNKNOWN", 0, "var(--text-secondary)", "UNKNOWN", 0.0, 0.0, 0.0)

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
    
    if not state["calibrated"]:
        detector.start_calibration()
    
    while cap.isOpened() and state["detection_active"]:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Check queue
        try:
            while True:
                cmd = st.session_state.command_queue.get_nowait()
                action = cmd.get("action")
                st.session_state.session_logger.log_event("VOICE_COMMAND", {"action": action})
                if action == "STOP":
                    state["detection_active"] = False
                    st.session_state.session_logger.end_session()
                elif action == "NOVA_STATUS":
                    st.session_state.nova_status = cmd.get("status", "○ OFFLINE").upper()
        except Empty:
            pass
        
        # Sync dynamic settings
        config.STATIC_EAR_THRESHOLD = state.get("baseline_ear", 0.3) * (ear_threshold / 100.0)
        config.EAR_CONSECUTIVE_FRAMES = frame_threshold
        if state["calibrated"]:
            detector.calibrated_threshold = config.STATIC_EAR_THRESHOLD

        result = detector.process_frame(frame)
        
        # Handle state logic
        if result["state"] == "CALIBRATING":
            if detector.is_calibrated:
                state["baseline_ear"] = detector.baseline_ear
                state["calibrated"] = True
        else:
            state["current_ear"] = result["ear"]
            alert_triggered_now = False
            if result["eyes_closed"] or result["yawning"]:
                if time.time() - last_alert > 2:
                    state["alert_count"] += 1
                    play_alert_sound()
                    last_alert = time.time()
                    alert_triggered_now = True

            risk_level, _ = get_risk_level(state["current_ear"], state.get("baseline_ear", 0.3))
            st.session_state.session_logger.update_metrics(result, risk_level, alert_triggered_now)

        # Draw minimalistic overlay on frame
        h, w = frame.shape[:2]
        if result["face_detected"] and result.get("landmarks"):
            # Subtle eye landmarks
            for idx in detector.LEFT_EYE + detector.RIGHT_EYE:
                lm = result["landmarks"][idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 212, 255), -1) # Cyan tracking
        
        # Display Video via HTML wrapper technique
        frame_resized = cv2.resize(frame, (640, 480))
        ret_jpg, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if ret_jpg:
            import base64
            b64 = base64.b64encode(buffer).decode("utf-8")
            
            face_status = "FACE DETECTED" if result["face_detected"] else "NO FACE DETECTED"
            ear_status = f"EAR {result['ear']:.2f}" if result["face_detected"] else ""
            
            camera_html = f"""
            <div class='camera-wrapper'>
                <div class='camera-overlay-top-left'>● LIVE</div>
                <div class='camera-overlay-top-right'>AI MONITORING</div>
                <img src="data:image/jpeg;base64,{b64}" style="width:100%; display:block;" />
                <div class='camera-overlay-bottom-left'>{face_status}</div>
                <div class='camera-overlay-bottom-right'>{ear_status}</div>
            </div>
            """
            camera_placeholder.markdown(camera_html, unsafe_allow_html=True)
        
        # Update UI Metrics (throttled)
        if frame_count % 5 == 0:
            if not state["calibrated"]:
                render_metrics("--", "--", "var(--warning)", 0, "WAIT", "var(--text-secondary)")
                render_safety_state("CALIBRATING", "Establish your baseline before monitoring. Look at the camera with eyes open.")
                render_side_panels("CALIBRATING", "var(--warning)", "UNKNOWN", 0, "var(--warning)", "UNKNOWN", result["ear"], result["mar"], 0.0)
            else:
                openness = int((state["current_ear"] / state["baseline_ear"]) * 100) if state["baseline_ear"] > 0 else 0
                openness = min(100, max(0, openness))
                risk_level, risk_color = get_risk_level(state["current_ear"], state["baseline_ear"])
                
                cnn_status = "READY" if result["cnn_available"] else "N/A"
                cnn_color = "var(--success)" if result["cnn_available"] else "var(--text-secondary)"
                
                render_metrics(openness, risk_level, risk_color, state["alert_count"], cnn_status, cnn_color)
                
                if result["eyes_closed"]:
                    render_safety_state("DROWSY", "Please pay attention to the road. Consider taking a break if necessary.")
                    driver_state, dcolor = "DROWSY", "var(--danger)"
                    rpct, rcolor = 100, "var(--danger)"
                elif openness < 80:
                    render_safety_state("WARNING", "Signs of reduced alertness detected.")
                    driver_state, dcolor = "WARNING", "var(--warning)"
                    rpct, rcolor = 60, "var(--warning)"
                else:
                    render_safety_state("ALERT", "Everything looks normal. Continue monitoring.")
                    driver_state, dcolor = "ALERT", "var(--success)"
                    rpct, rcolor = 10, "var(--success)"
                    
                cnn_state = result.get("cnn_eye_state", "UNKNOWN")
                cnn_conf = max(result.get("left_cnn_confidence", 0.0), result.get("right_cnn_confidence", 0.0))
                
                render_side_panels(driver_state, dcolor, risk_level, rpct, rcolor, cnn_state, result["ear"], result["mar"], cnn_conf)
                
            # Sync state dict for VoiceAssistant
            dur_str = "00:00:00"
            if state["session_start"]:
                dur_str = format_duration(int(time.time() - state["session_start"]))
                    
            st.session_state.app_state_dict.update({
                "monitoring": state["detection_active"],
                "risk_level": risk_level if state["calibrated"] else "Unknown",
                "drowsiness": result.get("eyes_closed", False),
                "alert_count": state["alert_count"],
                "session_duration": dur_str
            })
    
    detector.close()
    if cap:
        cap.release()
    st.session_state.session_logger.end_session()
else:
    import time
    time.sleep(0.5)
    st.rerun()
