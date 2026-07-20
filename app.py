import logging
logging.getLogger('streamlit').setLevel(logging.WARNING)

import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import pygame
import time
import streamlit as st
import threading
import streamlit.components.v1 as components
import os

# ── CNN Model Loading ───────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    """Load the pre-trained CNN model for eye state classification."""
    try:
        from tensorflow.keras.models import load_model
        model_path = "eye_state_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            logging.info("CNN model loaded successfully")
            return model
        else:
            logging.warning(f"Model file not found: {model_path}")
            st.error("CNN model file not found. Using geometric detection only.")
            return None
    except ImportError:
        logging.warning("TensorFlow not installed. CNN features unavailable.")
        st.warning("TensorFlow not installed. Using geometric detection only.")
        return None
    except Exception as e:
        logging.error(f"Error loading CNN model: {e}")
        st.error(f"Error loading CNN model: {e}")
        return None

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
        "current_ear":      0.32,
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
ASSISTANT_NAME = "Nova"

# Voice Assistant Init
if "voice_assistant" not in st.session_state:
    try:
        from voice_assistant import VoiceAssistant
        st.session_state.voice_assistant = VoiceAssistant(state)
    except Exception as e:
        logging.error(f"Voice assistant error: {e}")

# CNN Model Init
if "cnn_model" not in st.session_state:
    st.session_state.cnn_model = load_cnn_model()

# ── Audio setup ───────────────────────────────────────────
pygame.mixer.init()

def play_alert():
    sample_rate = 44100
    t     = np.linspace(0, 0.5, int(sample_rate * 0.5))
    wave  = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
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
        EAR_THRESHOLD = st.slider("EAR Threshold", 0.15, 0.35, 0.25)
        MAR_THRESHOLD = st.slider("MAR Threshold", 0.50, 0.90, 0.75)
        FRAME_THRESH = st.slider("Alert Sensitivity (Frames)", 10, 30, 20)
    
    with st.expander("ℹ️ Project Info"):
        st.markdown("Developer: Yashvanth K")
        st.markdown("CNN Accuracy: 98.45%")
        st.markdown("Dataset: MRL Eye Dataset (84K images)")
        st.markdown("Method: EAR + MAR + CNN")
        if st.button("Change to drowsy_detect.py mode", width='stretch'):
            pass

# ── Header ────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown("<h3>🚗 DrowsyGuard <span style='font-size:0.5em; color:#a0a0a0;'>| Powered by CNN 98.45%</span></h3>", unsafe_allow_html=True)
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
        c1, c2, c3, c4 = st.columns(4)
        start_btn = c1.button("▶ Start", width='stretch')
        pause_btn = c2.button("⏸ Pause", width='stretch')
        stop_btn  = c3.button("⏹ Stop", width='stretch')
        reset_btn = c4.button("🔄 Reset", width='stretch')

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

# ── MediaPipe Setup ───────────────────────────────────────
def eye_aspect_ratio(eye_points, landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    A = distance.euclidean(pt(eye_points[1]), pt(eye_points[5]))
    B = distance.euclidean(pt(eye_points[2]), pt(eye_points[4]))
    C = distance.euclidean(pt(eye_points[0]), pt(eye_points[3]))
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_points, landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    A = distance.euclidean(pt(mouth_points[1]), pt(mouth_points[7]))
    B = distance.euclidean(pt(mouth_points[2]), pt(mouth_points[6]))
    C = distance.euclidean(pt(mouth_points[3]), pt(mouth_points[5]))
    D = distance.euclidean(pt(mouth_points[0]), pt(mouth_points[4]))
    return (A + B + C) / (2.0 * D)

def extract_eye_region(frame, eye_points, landmarks, w, h, padding=10):
    """Extract eye region from frame using MediaPipe landmarks."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    
    # Get all eye landmark points
    points = np.array([pt(idx) for idx in eye_points])
    
    # Calculate bounding box with padding
    x_min = int(max(0, np.min(points[:, 0]) - padding))
    x_max = int(min(w, np.max(points[:, 0]) + padding))
    y_min = int(max(0, np.min(points[:, 1]) - padding))
    y_max = int(min(h, np.max(points[:, 1]) + padding))
    
    # Extract eye region
    eye_region = frame[y_min:y_max, x_min:x_max]
    
    return eye_region

def preprocess_eye_for_cnn(eye_region):
    """Preprocess eye region for CNN input (24x24 grayscale normalized)."""
    try:
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            eye_gray = eye_region
        
        # Resize to 24x24
        eye_resized = cv2.resize(eye_gray, (24, 24))
        
        # Normalize to [0, 1]
        eye_normalized = eye_resized.astype('float32') / 255.0
        
        # Reshape to (1, 24, 24, 1) for CNN input
        eye_input = eye_normalized.reshape(1, 24, 24, 1)
        
        return eye_input
    except Exception as e:
        logging.error(f"Error preprocessing eye region: {e}")
        return None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 39, 269, 405, 291, 375, 321, 308]
YAWN_THRESH = 15

# ── Camera Loop ───────────────────────────────────────────
if state.get("detection_active"):
    if state["session_start"] is None:
        state["session_start"] = time.time()
        
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    ear_counter = 0
    yawn_counter = 0
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
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        status_text = "ALERT"
        alert = False
        avg_ear = 0.0
        mar = 0.0
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            left_ear  = eye_aspect_ratio(LEFT_EYE,  lm, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, lm, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0
            mar       = mouth_aspect_ratio(MOUTH, lm, w, h)
            
            state["current_ear"] = avg_ear
            state["current_mar"] = mar
            
            # CNN Prediction (if model available)
            left_cnn_conf = 0.0
            right_cnn_conf = 0.0
            cnn_alert = False
            
            if st.session_state.get("cnn_model") is not None:
                try:
                    # Extract eye regions
                    left_eye_region = extract_eye_region(frame, LEFT_EYE, lm, w, h)
                    right_eye_region = extract_eye_region(frame, RIGHT_EYE, lm, w, h)
                    
                    # Preprocess for CNN
                    left_eye_input = preprocess_eye_for_cnn(left_eye_region)
                    right_eye_input = preprocess_eye_for_cnn(right_eye_region)
                    
                    # Run CNN predictions
                    if left_eye_input is not None and right_eye_input is not None:
                        cnn_model = st.session_state.cnn_model
                        left_pred = cnn_model.predict(left_eye_input, verbose=0)[0]
                        right_pred = cnn_model.predict(right_eye_input, verbose=0)[0]
                        
                        # Get confidence for "closed" class (index 1)
                        left_cnn_conf = left_pred[1] * 100  # Convert to percentage
                        right_cnn_conf = right_pred[1] * 100
                        
                        # Store in state for UI display
                        state["left_cnn_conf"] = left_cnn_conf
                        state["right_cnn_conf"] = right_cnn_conf
                        
                        # CNN alert if both eyes have high closed confidence
                        if left_cnn_conf > 70 and right_cnn_conf > 70:
                            cnn_alert = True
                except Exception as e:
                    logging.error(f"CNN prediction error: {e}")
                    state["left_cnn_conf"] = 0.0
                    state["right_cnn_conf"] = 0.0
            else:
                state["left_cnn_conf"] = 0.0
                state["right_cnn_conf"] = 0.0
            
            # Draw landmarks gently
            for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 212, 255), -1)
                
            # Hybrid decision logic: trigger alert if EAR threshold OR CNN detects closed eyes
            if avg_ear < EAR_THRESHOLD or cnn_alert:
                ear_counter += 1
                if ear_counter >= FRAME_THRESH:
                    status_text = "DROWSY — EYES CLOSED!"
                    alert = True
            else:
                ear_counter = 0
                
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_THRESH:
                    status_text = "DROWSY — YAWNING!"
                    alert = True
            else:
                yawn_counter = 0
                
            if state.get("dismissed"):
                alert = False
                state["dismissed"] = False
                
            if alert:
                if time.time() - last_alert > 2:
                    state["drowsy_count"] += 1
                    play_alert()
                    last_alert = time.time()
                    if "voice_assistant" in st.session_state:
                        st.session_state.voice_assistant._add_chat_message("system", "🔴 Drowsy alert triggered")
                        render_chat()
                    
            color = (44, 44, 255) if alert else (136, 255, 0)
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
            risk_label, risk_class = get_risk_level(avg_ear) if results.multi_face_landmarks else ("Unknown", "status-warn")
            
            # Get CNN confidence scores from state
            left_cnn = state.get("left_cnn_conf", 0.0)
            right_cnn = state.get("right_cnn_conf", 0.0)
            
            with metrics_container.container():
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                ear_color = "status-drowsy" if avg_ear < EAR_THRESHOLD and avg_ear > 0 else "status-alert"
                mar_color = "status-drowsy" if mar > MAR_THRESHOLD else "status-alert"
                cnn_left_color = "status-drowsy" if left_cnn > 70 else "status-alert"
                cnn_right_color = "status-drowsy" if right_cnn > 70 else "status-alert"
                
                m1.markdown(f"<div class='metric-card'><div class='metric-label'>👁 EAR</div><div class='metric-value {ear_color}'>{avg_ear:.3f}</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-card'><div class='metric-label'>👄 MAR</div><div class='metric-value {mar_color}'>{mar:.3f}</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='metric-label'>👁 CNN L</div><div class='metric-value {cnn_left_color}'>{left_cnn:.0f}%</div></div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-card'><div class='metric-label'>👁 CNN R</div><div class='metric-value {cnn_right_color}'>{right_cnn:.0f}%</div></div>", unsafe_allow_html=True)
                m5.markdown(f"<div class='metric-card'><div class='metric-label'>⚠️ Risk</div><div class='metric-value {risk_class}'>{risk_label}</div></div>", unsafe_allow_html=True)
                m6.markdown(f"<div class='metric-card'><div class='metric-label'>😴 Alerts</div><div class='metric-value'>{state['drowsy_count']}</div></div>", unsafe_allow_html=True)
                
            if not results.multi_face_landmarks:
                alert_banner_container.markdown(f"<div class='alert-warning'>⚠️ Please face the camera. Nova cannot see you.</div>", unsafe_allow_html=True)
            elif alert:
                alert_banner_container.markdown(f"<div class='alert-danger'>🚨 {status_text} — Wake up!</div>", unsafe_allow_html=True)
            elif ear_counter > FRAME_THRESH // 2 or yawn_counter > YAWN_THRESH // 2:
                alert_banner_container.markdown(f"<div class='alert-warning'>⚠️ Fatigue building... stay focused.</div>", unsafe_allow_html=True)
            else:
                alert_banner_container.markdown(f"<div class='alert-safe'>✅ Driver is alert and focused.</div>", unsafe_allow_html=True)
        
    if cap:
        cap.release()