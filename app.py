import logging
logging.getLogger('streamlit').setLevel(logging.WARNING)

import cv2
import numpy as np
import pygame
import time
import streamlit as st
import threading
import streamlit.components.v1 as components
import os
import config

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard — AI Driver Safety",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit defaults */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1rem;}

/* App background */
.stApp {
    background: linear-gradient(135deg, #080818 0%, #0d1117 50%, #1a1a2e 100%);
    color: #e0e0e0;
}

/* Glassmorphism card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
}

/* Metric cards */
.metric-card {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Status colors */
.status-alert  { color: #00ff88; }
.status-drowsy { color: #ff3344; }
.status-warn   { color: #ff8800; }

/* Pulsing dot */
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,255,136,0.7); }
    70%  { box-shadow: 0 0 0 10px rgba(0,255,136,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,255,136,0); }
}
.pulse-dot {
    width: 10px; height: 10px;
    background: #00ff88;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}

/* Alert banner */
.alert-safe     { background: rgba(0,255,136,0.1); border-left: 4px solid #00ff88; border-radius: 8px; padding: 12px 20px; margin-bottom: 20px; }
.alert-warning  { background: rgba(255,136,0,0.1); border-left: 4px solid #ff8800; border-radius: 8px; padding: 12px 20px; margin-bottom: 20px; }
.alert-danger   { background: rgba(255,51,68,0.15); border-left: 4px solid #ff3344; border-radius: 8px; padding: 12px 20px; margin-bottom: 20px; animation: blink 1s infinite; }
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.6; } }

/* Chat bubbles */
.chat-container {
    height: 380px;
    overflow-y: auto;
    padding: 12px;
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.bubble-user {
    background: rgba(0,212,255,0.15);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 6px 0 6px 40px;
    font-size: 14px;
    color: #00d4ff;
}
.bubble-nova {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px;
    margin: 6px 40px 6px 0;
    font-size: 14px;
    color: #e0e0e0;
}
.bubble-system {
    text-align: center;
    font-size: 11px;
    color: #555;
    margin: 4px 0;
}
.chat-time {
    font-size: 10px;
    color: #555;
    margin-top: 2px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0088cc);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,212,255,0.4);
}

/* Mic button special */
div[data-testid="stButton"] > button:has(div:contains("Speak to Nova")) {
    background: linear-gradient(135deg, #ff3344, #cc0022) !important;
    font-size: 16px !important;
    padding: 14px !important;
}

/* Sidebar */
.css-1d391kg { background: rgba(0,0,0,0.4); }

/* Custom Image Rounding */
[data-testid="stImage"] img {
    border-radius: 16px;
}
</style>
<script>
function scrollToBottom() {
    var container = document.getElementById('chat-scroll-target');
    if(container) {
        container.scrollTop = container.scrollHeight;
    }
}
setInterval(scrollToBottom, 500);
</script>
""", unsafe_allow_html=True)

# ── Shared State ──────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = {
        "detection_active": False,
        "paused":           False,
        "dismissed":        False,
        "pause_end":        0,
        "current_ear":      config.DEFAULT_BASELINE_EAR,
        "current_mar":      0.45,
        "session_start":    None,
        "drowsy_count":     0,
        "nova_status":      "👂 Listening for Nova...",
        "voice_log":        [],
        "last_heard":       "",
        "left_cnn_conf":    0.0,
        "right_cnn_conf":   0.0
    }

state = st.session_state.state
ASSISTANT_NAME = config.ASSISTANT_NAME

# Voice Assistant Init
if "voice_assistant" not in st.session_state:
    try:
        from voice_assistant import VoiceAssistant
        st.session_state.voice_assistant = VoiceAssistant(state)
    except Exception as e:
        logging.error(f"Voice assistant error: {e}")

# Detector Init
if "detector" not in st.session_state:
    from core.detector import DrowsyDetector
    st.session_state.detector = DrowsyDetector()

detector = st.session_state.detector

# ── Audio setup ───────────────────────────────────────────
pygame.mixer.init()

def play_alert():
    t     = np.linspace(0, config.AUDIO_DURATION, int(config.AUDIO_SAMPLE_RATE * config.AUDIO_DURATION))
    wave  = (np.sin(2 * np.pi * config.AUDIO_FREQ * t) * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
    sound.play()

# ── Metrics Helper ────────────────────────────────────────
def get_risk_level(ear):
    if ear > 0.30: return "Low", "status-alert"
    elif ear > 0.25: return "Medium", "text-electric"
    elif ear > 0.20: return "High", "status-warn"
    else: return "Critical", "status-drowsy"

def format_duration(start_time):
    if not start_time: return "00:00"
    elapsed = int(time.time() - start_time)
    mins = elapsed // 60
    secs = elapsed % 60
    return f"{mins:02d}:{secs:02d}"

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h2>🤖 Nova</h2>", unsafe_allow_html=True)
    st.markdown("AI Driver Safety Assistant", unsafe_allow_html=True)
    st.markdown("<div><span class='pulse-dot'></span> Active</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 🎙️ Voice Commands")
    st.markdown("""
    "Nova start" — Begin monitoring
    "Nova stop" — End session  
    "Nova pause" — Pause 2 mins
    "Nova resume" — Resume
    "Nova I'm fine" — Dismiss alert
    "Nova status" — Check fatigue
    "Nova how long" — Drive duration
    "Nova break" — Rest reminder
    "Nova breathing" — Breathing exercise
    "Nova water" — Hydration tip
    "Nova focus" — Focus tip
    "Nova emergency" — Emergency help
    """)
    
    with st.expander("⚙️ Settings"):
        st.markdown("*Settings now managed by config.py and calibration.*")
    
    with st.expander("ℹ️ Project Info"):
        st.markdown("Developer: Yashvanth K")
        st.markdown("Method: Core DrowsyDetector (EAR + MAR + CNN)")

# ── Header ────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    cnn_status = "Powered by CNN" if detector.cnn_available else "Geometric Detection"
    st.markdown(f"<h3>🚗 DrowsyGuard <span style='font-size:0.5em; color:#a0a0a0;'>| {cnn_status}</span></h3>", unsafe_allow_html=True)
with h_col2:
    timer = format_duration(state.get("session_start"))
    st.markdown(f"<div style='text-align: right; margin-top: 5px;'><span class='pulse-dot'></span> <b>Nova Active</b> | ⏱️ Session: {timer} | 🔔 {state['drowsy_count']} alerts</div>", unsafe_allow_html=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    video_container = st.empty()
    
    metrics_container = st.empty()
    # Render initial metrics
    with metrics_container.container():
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.markdown(f"<div class='metric-card'><div class='metric-label'>👁 EAR</div><div class='metric-value'>0.00</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-label'>👄 MAR</div><div class='metric-value'>0.00</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-label'>👁 CNN L</div><div class='metric-value'>0%</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-label'>👁 CNN R</div><div class='metric-value'>0%</div></div>", unsafe_allow_html=True)
        m5.markdown(f"<div class='metric-card'><div class='metric-label'>⚠️ Risk</div><div class='metric-value status-alert'>None</div></div>", unsafe_allow_html=True)
        m6.markdown(f"<div class='metric-card'><div class='metric-label'>😴 Alerts</div><div class='metric-value'>0</div></div>", unsafe_allow_html=True)
    
    alert_banner_container = st.empty()
    alert_banner_container.markdown(f"<div class='alert-safe'>✅ Driver is alert and focused.</div>", unsafe_allow_html=True)
    
    control_container = st.empty()
    with control_container.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        start_btn = c1.button("▶ Start", width='stretch')
        pause_btn = c2.button("⏸ Pause", width='stretch')
        stop_btn  = c3.button("⏹ Stop", width='stretch')
        reset_btn = c4.button("🔄 Reset", width='stretch')
        calib_btn = c5.button("🎯 Calibrate", width='stretch')

with right_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>🤖 Nova</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a0a0a0;'>Your AI driving assistant</p>", unsafe_allow_html=True)
    nova_status_box = st.empty()
    nova_status_box.markdown(f"<div style='font-size: 1.1rem; color: #00d4ff; margin-bottom: 10px;'><span class='pulse-dot'></span> <b>{state.get('nova_status', '👂 Listening...')}</b></div>", unsafe_allow_html=True)
    
    chat_container = st.empty()
    
    def render_chat():
        html = "<div class='chat-container' id='chat-scroll-target'>"
        for msg in state.get("voice_log", []):
            if msg["role"] == "user":
                html += f"<div class='bubble-user'>{msg['text']}<div class='chat-time'>{msg['time']}</div></div>"
            elif msg["role"] == "nova":
                html += f"<div class='bubble-nova'>🤖 {msg['text']}<div class='chat-time'>{msg['time']}</div></div>"
            elif msg["role"] == "system":
                html += f"<div class='bubble-system'>{msg['text']} <div class='chat-time'>{msg['time']}</div></div>"
        html += "</div>"
        chat_container.markdown(html, unsafe_allow_html=True)
        
    render_chat()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br><b>Quick Commands</b>", unsafe_allow_html=True)
    qc1, qc2, qc3 = st.columns(3)
    if qc1.button("▶ Start", width='stretch', key="qc_start"):
        if "voice_assistant" in st.session_state: st.session_state.voice_assistant._process_command("start"); render_chat()
    if qc2.button("⏸ Pause", width='stretch', key="qc_pause"):
        if "voice_assistant" in st.session_state: st.session_state.voice_assistant._process_command("pause"); render_chat()
    if qc3.button("✅ I'm fine", width='stretch', key="qc_fine"):
        if "voice_assistant" in st.session_state: st.session_state.voice_assistant._process_command("fine"); render_chat()
        
    qc4, qc5, qc6 = st.columns(3)
    if qc4.button("📊 Status", width='stretch', key="qc_status"):
        if "voice_assistant" in st.session_state: st.session_state.voice_assistant._process_command("status"); render_chat()
    if qc5.button("💧 Water", width='stretch', key="qc_water"):
        if "voice_assistant" in st.session_state: st.session_state.voice_assistant._process_command("water"); render_chat()
    if qc6.button("🆘 Emergency", width='stretch', key="qc_emergency"):
        if "voice_assistant" in st.session_state: st.session_state.voice_assistant._process_command("emergency"); render_chat()
        
    st.markdown("</div>", unsafe_allow_html=True)

# ── Logic Actions ─────────────────────────────────────────
if start_btn: state["detection_active"] = True
if pause_btn: 
    state["paused"] = True
    state["pause_end"] = time.time() + 120
if stop_btn: state["detection_active"] = False
if reset_btn:
    state["drowsy_count"] = 0
    state["session_start"] = None
    detector.reset_calibration()
if calib_btn:
    detector.start_calibration()
    state["detection_active"] = True  # Ensure camera is running for calibration

# ── Camera Loop ───────────────────────────────────────────
if state.get("detection_active"):
    if state["session_start"] is None:
        state["session_start"] = time.time()
        
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    last_alert = 0
    frame_count = 0
    
    while cap.isOpened() and state.get("detection_active"):
        
        # Handle pause
        if state.get("paused") and time.time() < state.get("pause_end", 0):
            alert_banner_container.markdown(f"<div class='alert-warning'>⏸️ Alerts paused by Nova — resume driving safely.</div>", unsafe_allow_html=True)
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (480, 360))
                ret_jpg, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if ret_jpg:
                    video_container.image(buffer.tobytes(), channels="BGR", width='stretch')
            continue
        elif state.get("paused"):
            state["paused"] = False
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        result = detector.process_frame(frame)
        
        h, w = frame.shape[:2]
        
        status_text = "NORMAL"
        alert = False
        
        avg_ear = result["ear"]
        mar = result["mar"]
        state["current_ear"] = avg_ear
        state["current_mar"] = mar
        
        if result["face_detected"]:
            if result["eyes_closed"]:
                status_text = "DROWSY — EYES CLOSED!"
                alert = True
            elif result["yawning"]:
                status_text = "DROWSY — YAWNING!"
                alert = True
                
            if state.get("dismissed"):
                alert = False
                state["dismissed"] = False
                
            if alert:
                if time.time() - last_alert > config.ALERT_COOLDOWN_SECONDS:
                    state["drowsy_count"] += 1
                    play_alert()
                    last_alert = time.time()
                    if "voice_assistant" in st.session_state:
                        st.session_state.voice_assistant._add_chat_message("system", "🔴 Drowsy alert triggered")
                        render_chat()
                        
            color = (44, 44, 255) if alert else (136, 255, 0)
            if alert:
                cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)
                cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
                cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        else:
            status_text = "NO FACE DETECTED"
            cv2.putText(frame, status_text, (w//2 - 100, h//2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
            
        # Update video every frame for smoother playback
        frame_resized = cv2.resize(frame, (480, 360))
        ret_jpg, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret_jpg:
            video_container.image(buffer.tobytes(), channels="BGR", width='stretch')
            
        # Update UI components less frequently to prevent Streamlit lag
        if frame_count % 5 == 0:
            risk_label, risk_class = get_risk_level(avg_ear) if result["face_detected"] else ("Unknown", "status-warn")
            
            left_cnn = result["left_cnn_confidence"]
            right_cnn = result["right_cnn_confidence"]
            
            with metrics_container.container():
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                ear_color = "status-drowsy" if avg_ear < detector.calibrated_threshold and avg_ear > 0 else "status-alert"
                mar_color = "status-drowsy" if mar > config.MAR_THRESHOLD else "status-alert"
                cnn_left_color = "status-drowsy" if left_cnn > config.CNN_CONFIDENCE_THRESHOLD else "status-alert"
                cnn_right_color = "status-drowsy" if right_cnn > config.CNN_CONFIDENCE_THRESHOLD else "status-alert"
                
                m1.markdown(f"<div class='metric-card'><div class='metric-label'>👁 EAR</div><div class='metric-value {ear_color}'>{avg_ear:.3f}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-label'>👄 MAR</div><div class='metric-value {mar_color}'>{mar:.3f}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-label'>👁 CNN L</div><div class='metric-value {cnn_left_color}'>{left_cnn:.0f}%</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='metric-label'>👁 CNN R</div><div class='metric-value {cnn_right_color}'>{right_cnn:.0f}%</div></div>", unsafe_allow_html=True)
                m5.markdown(f"<div class='metric-card'><div class='metric-label'>⚠️ Risk</div><div class='metric-value {risk_class}'>{risk_label}</div></div>", unsafe_allow_html=True)
                m6.markdown(f"<div class='metric-card'><div class='metric-label'>😴 Alerts</div><div class='metric-value'>{state['drowsy_count']}</div></div>", unsafe_allow_html=True)
                
            if not result["face_detected"]:
                alert_banner_container.markdown(f"<div class='alert-warning'>⚠️ Please face the camera. Nova cannot see you.</div>", unsafe_allow_html=True)
            elif result["state"] == "CALIBRATING":
                alert_banner_container.markdown(f"<div class='alert-warning'>🎯 Calibrating baseline EAR... Please look at the camera.</div>", unsafe_allow_html=True)
            elif alert:
                alert_banner_container.markdown(f"<div class='alert-danger'>🚨 {status_text} — Wake up!</div>", unsafe_allow_html=True)
            elif result["closed_frames"] > config.EAR_CONSECUTIVE_FRAMES // 2 or result["yawn_frames"] > config.MAR_CONSECUTIVE_FRAMES // 2:
                alert_banner_container.markdown(f"<div class='alert-warning'>⚠️ Fatigue building... stay focused.</div>", unsafe_allow_html=True)
            else:
                alert_banner_container.markdown(f"<div class='alert-safe'>✅ Driver is alert and focused.</div>", unsafe_allow_html=True)
        
    if cap:
        cap.release()