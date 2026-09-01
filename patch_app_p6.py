import re

with open("app_final.py", "r", encoding="utf-8") as f:
    content = f.read()

# Chunk 1: Session State Setup
chunk1_regex = re.compile(r"# ── Session State ──────────────────────────────────────────.*?from core\.detector import DrowsyDetector.*?import config", re.DOTALL)
chunk1_replacement = """# ── Session State ──────────────────────────────────────────
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
    st.session_state.voice_assistant.start()"""

content = chunk1_regex.sub(chunk1_replacement, content)


# Chunk 2: Sidebar
chunk2_regex = re.compile(r"ear_threshold = st\.slider\(\"Alert Threshold \(% of baseline\)\", 50, 90, 70\)\s+frame_threshold = st\.slider\(\"Alert Sensitivity \(frames\)\", 5, 30, 15\)")
chunk2_replacement = """ear_threshold = st.slider("Alert Threshold (% of baseline)", 50, 90, 70)
    frame_threshold = st.slider("Alert Sensitivity (frames)", 5, 30, 15)
    
    st.markdown("#### 🎙️ Voice Assistant")
    st.markdown(f"**Nova:** {st.session_state.nova_status}")"""

content = chunk2_regex.sub(chunk2_replacement, content)


# Chunk 3: Control Logic
chunk3_regex = re.compile(r"# ── Control Logic ─────────────────────────────────────────.*?if calibrate_btn:\s+state\[\"calibrated\"\] = False", re.DOTALL)
chunk3_replacement = """# ── Control Logic ─────────────────────────────────────────
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
    pass"""

content = chunk3_regex.sub(chunk3_replacement, content)


# Chunk 4: Detection Loop
chunk4_regex = re.compile(r"frame_count \+= 1")
chunk4_replacement = """frame_count += 1
        
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
            pass"""

content = chunk4_regex.sub(chunk4_replacement, content)


# Chunk 5: Sync state dict at end of frame
chunk5_regex = re.compile(r"if result\[\"eyes_closed\"\]:.*?alert_container\.markdown\(\"<div class='alert-safe'>✅ Driver alert and focused\.</div>\", unsafe_allow_html=True\)", re.DOTALL)
chunk5_replacement = """if result["eyes_closed"]:
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
            })"""
content = chunk5_regex.sub(chunk5_replacement, content)


# Chunk 6: Idle Loop
chunk6_regex = re.compile(r"    if cap:\s+cap\.release\(\)")
chunk6_replacement = """    if cap:
        cap.release()
else:
    # Idle loop checking for voice commands when camera is off
    import time
    time.sleep(0.5)
    st.rerun()"""

content = chunk6_regex.sub(chunk6_replacement, content)


with open("app_final.py", "w", encoding="utf-8") as f:
    f.write(content)
