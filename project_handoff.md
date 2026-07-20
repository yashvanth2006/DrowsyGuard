# DrowsyGuard Project Handoff Document

## Project Overview

**DrowsyGuard** is an AI-powered driver safety application that detects driver drowsiness in real-time using computer vision and machine learning. The system monitors eye closure and yawning patterns to alert drivers when fatigue is detected, reducing the risk of accidents caused by drowsy driving.

### Core Purpose
- Prevent drowsy driving accidents through real-time fatigue detection
- Provide hands-free voice control for safe driver interaction
- Deliver immediate audio and visual alerts when drowsiness is detected

### Main Features
- **Real-time Drowsiness Detection**: Uses Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) via MediaPipe face landmarks
- **CNN Model Integration**: Trained on MRL Eye Dataset (84K images) with 98.45% accuracy for eye state classification
- **Voice Assistant ("Nova")**: Two-stage voice pipeline with wake word detection and command recognition
- **Dual Interface**: Streamlit web UI with modern glassmorphism design + standalone OpenCV application
- **Audio Alerts**: Procedural sound generation using pygame (no external audio files required)
- **Session Tracking**: Monitors driving duration, alert count, and fatigue risk levels
- **Hands-free Commands**: Voice control for start, stop, pause, status checks, and safety tips

---

## Tech Stack & Tooling

### Core Languages & Frameworks
- **Python 3.x**: Primary language
- **Streamlit 1.58.0**: Web UI framework
- **TensorFlow/Keras 2.13.1**: Deep learning for CNN model
- **OpenCV 5.0.0**: Computer vision and image processing

### Computer Vision & ML
- **MediaPipe 0.10.11**: Face mesh landmark detection (468 points)
- **NumPy 2.4.6**: Numerical computations
- **SciPy 1.17.1**: Spatial distance calculations for EAR/MAR
- **Scikit-learn 1.9.0**: Model evaluation metrics

### Voice & Audio
- **SpeechRecognition 3.17.0**: Google Speech-to-Text API
- **pyttsx3 2.99**: Text-to-Speech engine
- **PyAudio 0.2.14**: Microphone input
- **Pygame 2.6.1**: Audio alert generation
- **Vosk** (optional): Local wake word detection (if model present)

### Visualization & UI
- **Matplotlib 3.11.0**: Training curve visualization
- **Altair 6.2.2**: Data visualization (Streamlit dependency)

### Utilities
- **imutils 0.5.4**: OpenCV convenience functions
- **PyWin32 312**: Windows-specific APIs

---

## Architecture & File Structure

```
DrowsyGuard/
├── app.py                      # Main Streamlit web application (511 lines)
├── drowsy_detect.py            # Standalone OpenCV detection (142 lines)
├── voice_assistant.py          # Voice assistant logic (303 lines)
├── settings_manager.py         # Settings persistence (29 lines)
├── train_model.py              # CNN training script (114 lines)
├── test.py                     # Webcam testing utility (20 lines)
├── eye_state_model.h5          # Trained CNN model (1.8MB)
├── training_curves.png         # Model training visualization
├── assistant_settings.json     # Voice assistant configuration
├── mic_test.py                 # Microphone testing
├── mic_test.wav                # Audio test sample
├── .venv/                      # Virtual environment
├── archive/                    # Training data and utilities
│   ├── archive.zip             # Compressed dataset
│   └── data/
│       ├── train/              # Training images (awake/sleepy folders)
│       ├── val/                # Validation images
│       ├── test/               # Test images
│       ├── readme.md           # Dataset documentation
│       ├── labels.txt          # Class labels
│       ├── get_info.py         # Dataset info script
│       └── split_data.py       # Train/val/test splitter
└── project_handoff.md          # This document
```

### Component Responsibilities

**app.py** - Streamlit Web Interface
- Session state management
- MediaPipe face mesh integration
- Real-time video processing loop
- UI rendering (metrics, alerts, chat)
- Control button handlers
- Settings sidebar configuration

**drowsy_detect.py** - Standalone Detection
- Lightweight OpenCV-only version
- Direct webcam output
- No UI dependencies
- Ideal for embedded systems

**voice_assistant.py** - Voice Control System
- Two-stage recognition pipeline
- Wake word detection (Vosk + Google fallback)
- Command processing and response
- TTS queue management
- Breathing exercise guidance

**settings_manager.py** - Configuration
- JSON-based settings persistence
- Assistant name customization
- Voice speed and volume controls

**train_model.py** - Model Training
- CNN architecture definition
- Data loading and preprocessing
- Training loop with validation
- Model serialization to H5 format

---

## Current State (Completed)

### Fully Working Features

1. **Streamlit Web Interface**
   - Modern glassmorphism UI with custom CSS
   - Real-time video feed with face landmark overlay
   - Live metrics display (EAR, MAR, Risk Level, Alert Count)
   - Dynamic alert banners (safe/warning/danger)
   - Session timer and alert counter
   - Responsive layout with sidebar controls

2. **Drowsiness Detection Algorithm**
   - MediaPipe FaceMesh integration (468 landmark points)
   - Eye Aspect Ratio (EAR) calculation using 6 eye landmarks per eye
   - Mouth Aspect Ratio (MAR) calculation using 8 mouth landmarks
   - Configurable thresholds via UI sliders
   - Frame-based counter system to prevent false positives
   - Yawn detection with 15-frame threshold

3. **Voice Assistant ("Nova")**
   - Wake word detection: "nova", "no va", "nover", "over", "know va", "nola", "noa"
   - Two-stage pipeline: local wake word → Google command recognition
   - 12 voice commands: start, stop, pause, resume, fine, status, how long, break, breathing, water, focus, emergency
   - Real-time chat interface with message history
   - TTS responses with deduplication
   - Breathing exercise guided mode
   - Activation beep feedback

4. **Audio Alert System**
   - Procedural beep generation (1000Hz sine wave)
   - No external audio file dependencies
   - 2-second cooldown between alerts
   - Pygame mixer integration

5. **CNN Model**
   - Trained on MRL Eye Dataset (84K images)
   - Architecture: 2 Conv2D layers + MaxPooling + Dense + Dropout
   - Validation accuracy: 98.45%
   - Model saved as `eye_state_model.h5`
   - Training curves visualized and saved

6. **Session Management**
   - Driving duration tracking
   - Drowsy alert counter
   - Pause/resume functionality with 2-minute timer
   - Alert dismissal mechanism
   - Risk level categorization (Low/Medium/High/Critical)

7. **Settings System**
   - JSON-based persistence
   - Configurable EAR threshold (0.15-0.35)
   - Configurable MAR threshold (0.50-0.90)
   - Configurable alert sensitivity (10-30 frames)
   - Assistant name customization

---

## Work in Progress & Known Bugs

### Known Issues

1. **Streamlit Command Error**
   - **Issue**: User attempted `streamlit app.py` instead of `streamlit run app.py`
   - **Impact**: Application won't start with incorrect command
   - **Fix**: Use correct command: `streamlit run app.py`

2. **Vosk Model Dependency**
   - **Issue**: Vosk model folder `vosk-model-small-en-us` may not exist
   - **Impact**: Falls back to Google-only recognition (less efficient)
   - **Status**: Graceful fallback implemented, not critical

3. **Camera Resource Management**
   - **Issue**: Camera may not release properly on abrupt termination
   - **Impact**: Camera remains locked until process restart
   - **Mitigation**: Proper release in normal exit paths

4. **UI Update Frequency**
   - **Issue**: Metrics update every 5 frames to prevent Streamlit lag
   - **Impact**: Slight delay in metric updates
   - **Trade-off**: Necessary for performance

### Partially Implemented Features

1. **CNN Model Integration**
   - **Status**: Model trained and saved but not integrated into detection pipeline
   - **Current**: Detection uses only MediaPipe EAR/MAR
   - **Opportunity**: Combine CNN predictions with geometric features for hybrid approach

2. **Mode Switching**
   - **Status**: UI button exists for "Change to drowsy_detect.py mode" but non-functional
   - **Current**: Placeholder implementation
   - **Opportunity**: Implement actual mode switching or remove button

3. **Dataset Structure**
   - **Status**: Archive folder structure exists but train/val/test folders appear empty
   - **Current**: Dataset may be in archive.zip
   - **Action Required**: Extract and verify dataset if retraining needed

---

## Immediate Next Steps

### Priority 1: Fix Application Startup
- Document correct startup command: `streamlit run app.py`
- Add README with setup instructions
- Create batch script or alias for easy startup

### Priority 2: CNN Model Integration
- Load `eye_state_model.h5` in app.py
- Extract eye regions from video frames
- Pass to CNN for classification
- Combine CNN output with EAR/MAR for weighted decision
- Update UI to show CNN confidence scores

### Priority 3: Dataset Verification
- Extract archive.zip if needed
- Verify train/val/test folder structure
- Test train_model.py with current dataset
- Document dataset requirements for future training

### Priority 4: Mode Switching Implementation
- Either implement functional mode switching between Streamlit and OpenCV modes
- Or remove the placeholder button from UI

### Priority 5: Enhanced Error Handling
- Add camera initialization checks
- Graceful fallback if camera unavailable
- Voice assistant error recovery
- Network dependency handling for Google Speech API

---

## Environment & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or USB)
- Microphone (for voice commands)
- Windows OS (current deployment target)

### Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install streamlit opencv-python mediapipe numpy scipy pygame
pip install SpeechRecognition pyttsx3 pyaudio
pip install tensorflow scikit-learn matplotlib imutils
```

### Required Packages (with versions)
```
streamlit==1.58.0
opencv-python==5.0.0
mediapipe==0.10.11
numpy==2.4.6
scipy==1.17.1
pygame==2.6.1
SpeechRecognition==3.17.0
pyttsx3==2.99
pyaudio==0.2.14
tensorflow==2.13.1
scikit-learn==1.9.0
matplotlib==3.11.0
imutils==0.5.4
```

### Running the Application

**Streamlit Web UI (Primary)**
```powershell
streamlit run app.py
```

**Standalone OpenCV Version**
```powershell
python drowsy_detect.py
```

**Webcam Test**
```powershell
python test.py
```

**Model Training** (if dataset available)
```powershell
python train_model.py
```

### Environment Variables
No environment variables required. All configuration is handled through:
- UI sliders in Streamlit sidebar
- `assistant_settings.json` file
- Hardcoded constants in Python files

### Microphone Setup
- Ensure PyAudio is installed correctly on Windows
- Test microphone with `mic_test.py`
- Grant microphone permissions to Python/terminal

### Camera Setup
- Ensure camera drivers are installed
- Test with `test.py` script
- Camera index 0 is used by default (modify in code if multiple cameras)

---

## Critical Code Snippets

### MediaPipe Face Landmark Indices
```python
# Landmark indices for facial features
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 39, 269, 405, 291, 375, 321, 308]
```

### Eye Aspect Ratio (EAR) Calculation
```python
def eye_aspect_ratio(eye_points, landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    A = distance.euclidean(pt(eye_points[1]), pt(eye_points[5]))
    B = distance.euclidean(pt(eye_points[2]), pt(eye_points[4]))
    C = distance.euclidean(pt(eye_points[0]), pt(eye_points[3]))
    return (A + B) / (2.0 * C)
```

### Mouth Aspect Ratio (MAR) Calculation
```python
def mouth_aspect_ratio(mouth_points, landmarks, w, h):
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    A = distance.euclidean(pt(mouth_points[1]), pt(mouth_points[7]))
    B = distance.euclidean(pt(mouth_points[2]), pt(mouth_points[6]))
    C = distance.euclidean(pt(mouth_points[3]), pt(mouth_points[5]))
    D = distance.euclidean(pt(mouth_points[0]), pt(mouth_points[4]))
    return (A + B + C) / (2.0 * D)
```

### MediaPipe Initialization
```python
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Default Thresholds
```python
EAR_THRESHOLD    = 0.25  # Eyes closed below this value
MAR_THRESHOLD    = 0.75  # Yawning above this value
FRAME_THRESHOLD  = 20    # Consecutive frames for alert
YAWN_THRESHOLD   = 15    # Consecutive frames for yawn alert
```

### Voice Assistant Wake Words
```python
WAKE_WORDS = ["nova", "no va", "nover", "over", "know va", "nola", "noa"]
```

### Session State Structure
```python
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
    "last_heard":       ""
}
```

### CNN Model Architecture
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
```

### Audio Alert Generation
```python
def play_alert():
    sample_rate = 44100
    t = np.linspace(0, 0.5, int(sample_rate * 0.5))
    wave = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
    sound.play()
```

### Voice Command Processing
```python
def _process_command(self, command: str):
    c = command.lower()
    if any(w in c for w in ["start", "begin", "monitor", "go"]):
        self.state["detection_active"] = True
        self.state["session_start"] = time.time()
        response = "Monitoring started. Stay alert and drive safely."
    # ... additional command handlers
```

### Settings Configuration
```python
DEFAULT_SETTINGS = {
    "assistant_name": None,
    "voice_speed": "normal",
    "alert_volume": 1.0,
    "auto_start": False
}
```

---

## Additional Notes

### Performance Considerations
- Camera resolution set to 640x480 for real-time performance
- FPS limited to 30 for consistent processing
- UI updates throttled to every 5 frames to reduce Streamlit lag
- Frame buffer size set to 1 to minimize latency

### Security & Privacy
- All processing is local (no cloud data transmission except Google Speech API)
- No video/audio storage by default
- Voice assistant uses Google Speech API (requires internet)
- Consider offline Vosk model for fully local operation

### Testing Checklist
- [ ] Webcam initialization and video feed
- [ ] Face landmark detection accuracy
- [ ] EAR/MAR calculation correctness
- [ ] Alert triggering on drowsiness simulation
- [ ] Voice wake word detection
- [ ] Voice command recognition
- [ ] TTS audio output
- [ ] Audio alert generation
- [ ] Session state persistence
- [ ] Settings save/load functionality
- [ ] UI responsiveness on different screen sizes

### Contact Information
- **Developer**: Yashvanth K
- **Project**: DrowsyGuard
- **CNN Accuracy**: 98.45%
- **Dataset**: MRL Eye Dataset (84K images)
- **Method**: EAR + MAR + CNN (hybrid approach planned)

---

**Document Version**: 1.0  
**Last Updated**: 2026-07-20  
**Status**: Ready for handoff
