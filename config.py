import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# ── 1. Camera ───────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ── 2. Calibration ──────────────────────────────────────────
# 150 frames at 30 FPS = 5 seconds of calibration
CALIBRATION_FRAMES = 150
DEFAULT_BASELINE_EAR = 0.30
# Default percentage of baseline EAR to trigger an alert
CALIBRATION_EAR_THRESHOLD_PERCENT = 70.0

# ── 3. Eye Detection (Geometric) ────────────────────────────
# Static EAR threshold if calibration is not used
STATIC_EAR_THRESHOLD = 0.25
# Number of consecutive frames the EAR must be below threshold to trigger an alert
EAR_CONSECUTIVE_FRAMES = 15

# ── 4. Mouth / Yawning ──────────────────────────────────────
# Threshold for Mouth Aspect Ratio (MAR) to detect a yawn
MAR_THRESHOLD = 0.75
# Number of consecutive frames the MAR must be above threshold to trigger a yawn alert
MAR_CONSECUTIVE_FRAMES = 15

# ── 5. CNN ──────────────────────────────────────────────────
CNN_MODEL_PATH = BASE_DIR / "eye_state_model.h5"
CNN_INPUT_WIDTH = 24
CNN_INPUT_HEIGHT = 24
CNN_INPUT_CHANNELS = 1
# Confidence percentage required to classify an eye as closed (0 to 100)
CNN_CONFIDENCE_THRESHOLD = 70.0

# ── 6. Alerts ───────────────────────────────────────────────
AUDIO_ALERT_ENABLED = True
# Cooldown between consecutive audio alerts (in seconds)
ALERT_COOLDOWN_SECONDS = 2.0
# Audio generation settings
AUDIO_SAMPLE_RATE = 44100
AUDIO_FREQ = 1000
AUDIO_DURATION = 0.5

# ── 7. Voice Assistant ──────────────────────────────────────
VOICE_ENABLED = True
TTS_ENABLED = True

ASSISTANT_NAME = "nova"
WAKE_WORDS = ["nova", "no va", "nover", "over", "know va", "nola", "noa"]

# Speech recognition timing limits (in seconds)
WAKE_WORD_TIMEOUT = 10
WAKE_WORD_PHRASE_TIME_LIMIT = 2

COMMAND_TIMEOUT = 6
COMMAND_PHRASE_TIME_LIMIT = 5

# Text-to-Speech settings
TTS_RATE = 175
TTS_VOLUME = 1.0

# Local offline speech model path
VOSK_MODEL_PATH = BASE_DIR / "vosk-model-small-en-us"
