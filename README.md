# DrowsyGuard Pro 🚗👁️

DrowsyGuard Pro is a practical, real-time driver safety application that detects drowsiness using computer vision and MediaPipe face landmarks. Designed for real-world use with personalized calibration and reliable performance.

## ✨ Key Features

- **Personalized Calibration**: System learns your baseline eye openness for accurate detection
- **Real-time Detection**: Monitors eye closure using Eye Aspect Ratio (EAR)
- **Session Analytics**: Tracks driving sessions and alert history
- **Emergency Contact**: Quick access to emergency contact information
- **Offline-First**: Works without internet connection
- **Simple Controls**: Easy-to-use button interface (no voice commands needed)
- **Adjustable Sensitivity**: Customize alert threshold and sensitivity

## Prerequisites

- **OS**: Windows 10/11
- **Python**: 3.8 or higher
- **Webcam**: Built-in or external camera
- **Good Lighting**: Adequate lighting for accurate face detection

## Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd DrowsyGuard
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app_final.py
   ```

## Usage Guide

### First-Time Setup

1. **Launch the application** using the command above
2. **Allow camera access** when prompted
3. **Position yourself** in front of the camera with good lighting
4. **Click "Calibrate Baseline"** while keeping your eyes fully open
5. **Wait 5 seconds** while the system records your baseline eye openness
6. **Start Monitoring** by clicking the "Start Monitoring" button

### During Use

- **Monitor the metrics**: Eye openness percentage, risk level, and alert count
- **Adjust settings**: Use sidebar to change alert threshold and sensitivity
- **Emergency contact**: Set your emergency contact in the sidebar for quick access
- **Recalibrate**: Click "Recalibrate" if you add/remove glasses or lighting changes

### Settings Explained

- **Alert Threshold**: Percentage of baseline eye openness that triggers alerts (default: 70%)
- **Alert Sensitivity**: Number of consecutive frames below threshold before alert (default: 15)
- **Emergency Contact**: Phone number or contact for emergency situations

## How It Works

1. **MediaPipe Face Mesh** detects 468 facial landmarks
2. **Eye Aspect Ratio (EAR)** calculates eye openness using 6 points per eye
3. **Baseline Calibration** records your normal eye openness
4. **Real-time Monitoring** compares current EAR to your baseline
5. **Alert System** triggers when eyes remain closed for configured duration

## Troubleshooting

**Camera not working:**
- Ensure camera is not in use by another application
- Check camera permissions in Windows settings
- Try restarting the application

**Poor detection accuracy:**
- Improve lighting conditions (avoid backlighting)
- Position camera at eye level
- Recalibrate baseline
- Ensure face is clearly visible in frame

**False alerts:**
- Increase alert threshold percentage
- Increase alert sensitivity (frame count)
- Recalibrate with eyes fully open

**No alerts when drowsy:**
- Decrease alert threshold percentage
- Decrease alert sensitivity (frame count)
- Ensure calibration was done with eyes fully open

## Technical Details

- **Detection Method**: Eye Aspect Ratio (EAR) using MediaPipe Face Mesh
- **Frame Rate**: 30 FPS
- **Resolution**: 640x480
- **Dependencies**: Streamlit, OpenCV, MediaPipe, NumPy, SciPy, Pygame
- **No Internet Required**: Fully offline operation

## Project Structure

```
DrowsyGuard/
├── app_final.py          # Main application (FINAL VERSION)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .venv/               # Virtual environment
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) for face mesh detection
- [Streamlit](https://streamlit.io/) for the web interface framework