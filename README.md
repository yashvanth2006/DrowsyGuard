# DrowsyGuard Pro 🚗👁️

## 1. Project Overview
DrowsyGuard Pro is a practical, real-time driver safety application that detects drowsiness using computer vision and MediaPipe face landmarks. It features personalized calibration, robust analytics, and a voice assistant. Designed for real-world use with secure, offline-first processing.

## 2. Technology Stack
- **Frontend & UI**: Streamlit, OpenCV
- **Computer Vision**: MediaPipe, OpenCV
- **Deep Learning**: TensorFlow / Keras (CNN for eye state classification)
- **Voice Assistant**: SpeechRecognition, Vosk, pyttsx3
- **Audio Alerts**: Pygame
- **Data & Analytics**: Python stdlib (JSON)

## 3. Requirements
- **OS**: Windows 10/11 (Development and current primary support)
- **Python**: 3.11.x (Tested on 3.11.7)
- **Hardware**: 
  - Webcam / Camera for face detection
  - Microphone for Voice Assistant
  - Audio output (speakers/headphones) for alerts
- **Model Requirement**: A pre-trained CNN model `eye_state_model.h5` is required for hybrid detection.

## 4. Installation

1. **Navigate to the project directory:**
   ```bash
   cd DrowsyGuard
   ```

2. **Create and activate virtual environment (Windows):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## 5. MediaPipe Compatibility
**CRITICAL**: This project relies on `mediapipe==0.10.14`.
The application currently uses the legacy `mp.solutions` API (specifically `mp.solutions.face_mesh`). Upgrading MediaPipe past `0.10.14` may break face detection because newer versions deprecate the `solutions` API in favor of Tasks. Do not upgrade MediaPipe casually.

## 6. Model Setup
The hybrid CNN detection requires `eye_state_model.h5` to be placed in the root directory (`DrowsyGuard/eye_state_model.h5`). If the model is missing, the application will gracefully fall back to pure geometric (EAR) detection without crashing.

## 7. Configuration
Application settings are centralized in `config.py`. This includes camera settings, alert thresholds, voice assistant timeouts, and analytics flags. For environment-specific secrets, a `.env.example` is provided, though currently, no hardcoded secrets are required for the application to run.

## 8. Running Streamlit
To launch the main Streamlit interface:
```bash
streamlit run app_final.py
```

## 9. Running Standalone Detector
To launch the headless OpenCV standalone detector:
```bash
python drowsy_detect.py
```

## 10. Calibration
**Calibration Process**: When you start the application or click 'Calibrate', sit in front of the camera with your eyes fully open for 5 seconds. The system calculates your baseline Eye Aspect Ratio (EAR) and creates a personalized threshold. This ensures accuracy across different users and conditions (e.g., wearing glasses).

## 11. Keyboard Controls
When running the **Standalone Detector** (`drowsy_detect.py`), the OpenCV window supports:
- **Q** — Quit the application
- **C** — Calibrate baseline
- **R** — Reset calibration

## 12. Voice Assistant
DrowsyGuard Pro features **Nova**, an integrated voice assistant:
- **Vosk**: Local offline speech recognition used as the primary engine. Requires the `vosk-model-small-en-us` directory in the project root.
- **Google Fallback**: If Vosk is unavailable, Nova gracefully falls back to Google's online speech recognition.
- **Graceful Failure**: If a microphone is unavailable or voice libraries fail to load, the voice assistant disables itself without crashing the core drowsiness detection.

## 13. Analytics
- **Local Storage**: Session analytics are stored securely on the local machine in `data/sessions/`.
- **Session Data**: Tracks metrics such as maximum risk level, total alerts, drowsiness events, and session duration.
- **Privacy Model**: No cloud analytics are used by default (`CLOUD_ANALYTICS_ENABLED = False`). 

## 14. Troubleshooting
- **MediaPipe Errors**: Verify you are using exactly `mediapipe==0.10.14`.
- **Missing Model**: Place `eye_state_model.h5` in the root folder. Detection will still work via EAR fallback.
- **Camera Unavailable**: Ensure no other application (like Zoom/Teams) is using the camera.
- **Microphone Unavailable**: Check system sound settings; Nova will disable itself safely if missing.
- **Dependency Problems**: Run `python -m pip check` to ensure all packages in `requirements.txt` are correctly installed.

## 15. Security
- **Secrets Management**: Configuration uses environment variables where applicable. Do not commit `.env` files.
- **No Raw Media Persistence**: The application intentionally does **NOT** save raw camera frames, raw audio recordings, or raw speech transcripts to disk. 
- **Analytics**: Analytics are entirely local/offline. Unnecessary PII is not stored.

## 16. Hardware Limitations
Full functional validation of camera capturing, face detection, microphone input, and speaker output fundamentally requires the appropriate physical hardware. Automated software tests can verify the logic, but real-world physical verification is environment-dependent.

---

## PRODUCTION RELEASE CHECKLIST
- [x] Python environment verified
- [x] requirements.txt verified
- [x] MediaPipe 0.10.14 pinned
- [x] pip check passes
- [x] dependency audit completed where available (Note: pip-audit unavailable)
- [x] no hardcoded secrets
- [x] .gitignore verified
- [x] model artifact verified
- [x] Streamlit startup verified
- [x] standalone startup verified
- [x] tests pass
- [x] compileall passes
- [x] analytics privacy verified
- [x] camera cleanup verified
- [x] voice cleanup verified
- [x] Git working tree reviewed
- [x] README complete
- [x] hardware-dependent tests documented