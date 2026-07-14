# Two-stage voice pipeline
# Stage 1: Local wake word (Vosk or energy-based)
# Stage 2: Google API for command recognition

import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

ASSISTANT_NAME = "nova"
WAKE_WORDS = ["nova", "no va", "nover", "over", "know va", "nola", "noa"]

class VoiceAssistant:
    def __init__(self, state: dict):
        self.state          = state
        self.name           = ASSISTANT_NAME
        self.running        = True
        self.tts_queue      = queue.Queue()
        self.wake_detected  = False
        self.last_spoken    = {}         # for deduplication
        self.recognizer     = sr.Recognizer()
        self.command_recognizer = sr.Recognizer()
        
        # Stage 1 recognizer — aggressive settings for speed
        self.recognizer.energy_threshold         = 400
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold          = 0.5
        self.recognizer.phrase_threshold         = 0.3
        self.recognizer.non_speaking_duration    = 0.3
        
        # Stage 2 recognizer — balanced for accuracy
        self.command_recognizer.energy_threshold         = 300
        self.command_recognizer.dynamic_energy_threshold = False
        self.command_recognizer.pause_threshold          = 0.8
        
        # Try to load Vosk for local wake word
        self.vosk_model = None
        self._load_vosk()
        
        # Start threads
        threading.Thread(target=self._tts_thread,          daemon=True).start()
        threading.Thread(target=self._wake_word_pipeline,  daemon=True).start()
        
        logger.debug("VoiceAssistant initialized")
    
    def _load_vosk(self):
        try:
            from vosk import Model, KaldiRecognizer
            import os
            model_path = "vosk-model-small-en-us"
            if os.path.exists(model_path):
                self.vosk_model = Model(model_path)
                logger.debug("Vosk model loaded")
            else:
                logger.debug("Vosk model not found, using fallback")
        except ImportError:
            logger.debug("Vosk not installed, using fallback")
    
    def _tts_thread(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
        while self.running:
            try:
                text = self.tts_queue.get(timeout=2)
                # Deduplication
                now = time.time()
                if text in self.last_spoken:
                    if now - self.last_spoken[text] < 3:
                        continue
                self.last_spoken[text] = now
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"TTS error: {e}")
    
    def speak(self, text: str):
        self.tts_queue.put(text)
        logs = self.state.get("voice_log", [])
        logs.append({"role": "nova", "text": text, "time": time.strftime("%H:%M:%S")})
        self.state["voice_log"] = logs[-20:]

    def _add_chat_message(self, role: str, text: str):
        logs = self.state.get("voice_log", [])
        logs.append({"role": role, "text": text, "time": time.strftime("%H:%M:%S")})
        self.state["voice_log"] = logs[-20:]
    
    def _play_activation_beep(self):
        try:
            import pygame
            sample_rate = 44100
            t = np.linspace(0, 0.15, int(sample_rate * 0.15))
            wave = (np.sin(2 * np.pi * 880 * t) * 16383).astype(np.int16)
            stereo = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
        except Exception:
            pass
    
    def _is_wake_word(self, text: str) -> bool:
        text = text.lower().strip()
        return any(w in text for w in WAKE_WORDS)
    
    def _wake_word_pipeline(self):
        logger.debug(f"Wake word pipeline started for: {self.name}")
        
        while self.running:
            try:
                # ── STAGE 1: Local Wake Word Detection ──
                with sr.Microphone() as source:
                    # Short chunk — just enough to detect wake word
                    audio = self.recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=2   # max 2 seconds for wake word
                    )
                
                # Try local Vosk first
                wake_detected = False
                
                if self.vosk_model:
                    wake_detected = self._check_vosk(audio)
                
                # Fallback: quick Google check on short audio
                if not wake_detected:
                    try:
                        text = self.recognizer.recognize_google(
                            audio, show_all=False
                        ).lower()
                        logger.debug(f"Stage 1 heard: {text}")
                        self.state["last_heard"] = text
                        wake_detected = self._is_wake_word(text)
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        pass
                
                # ── STAGE 2: Command Recognition ──
                if wake_detected:
                    logger.debug("Wake word detected! Opening command window...")
                    self.state["nova_status"] = "🔊 Listening for command..."
                    self._play_activation_beep()
                    
                    # Record command (longer window)
                    try:
                        with sr.Microphone() as source:
                            command_audio = self.command_recognizer.listen(
                                source,
                                timeout=6,
                                phrase_time_limit=5
                            )
                        
                        command = self.command_recognizer.recognize_google(
                            command_audio
                        ).lower()
                        
                        logger.debug(f"Command recognized: {command}")
                        
                        # Log user speech to chat
                        logs = self.state.get("voice_log", [])
                        logs.append({
                            "role": "user",
                            "text": command,
                            "time": time.strftime("%H:%M:%S")
                        })
                        self.state["voice_log"] = logs[-20:]
                        self.state["nova_status"] = f"🎤 Heard: {command}"
                        
                        # Process command
                        self._process_command(command)
                        
                    except sr.UnknownValueError:
                        self.speak("I didn't catch that. Please try again.")
                        self.state["nova_status"] = "❓ Could not understand"
                    except sr.WaitTimeoutError:
                        self.state["nova_status"] = "⏱️ Command timeout"
                    except sr.RequestError:
                        self.state["nova_status"] = "❌ Recognition error"
                    
                    # Brief pause before returning to wake word detection
                    time.sleep(0.5)
                    self.state["nova_status"] = "👂 Listening for Nova..."
                    
            except sr.WaitTimeoutError:
                # No speech in timeout window — completely normal, loop again
                pass
            except Exception as e:
                logger.debug(f"Pipeline error: {e}")
                time.sleep(1)
    
    def _check_vosk(self, audio) -> bool:
        try:
            from vosk import KaldiRecognizer
            import json
            rec = KaldiRecognizer(self.vosk_model, 16000)
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            rec.AcceptWaveform(raw)
            result = json.loads(rec.Result())
            text = result.get("text", "").lower()
            logger.debug(f"Vosk heard: {text}")
            return self._is_wake_word(text)
        except Exception:
            return False
    
    def _process_command(self, command: str):
        c = command.lower()
        response = None
        
        if any(w in c for w in ["start", "begin", "monitor", "go"]):
            self.state["detection_active"] = True
            self.state["session_start"]    = time.time()
            self.state["drowsy_count"]     = 0
            response = "Monitoring started. Stay alert and drive safely."

        elif any(w in c for w in ["stop", "end", "finish"]):
            self.state["detection_active"] = False
            response = "Monitoring stopped. Drive safely."

        elif "pause" in c:
            self.state["paused"]    = True
            self.state["pause_end"] = time.time() + 120
            response = "Alerts paused for 2 minutes."

        elif "resume" in c:
            self.state["paused"] = False
            response = "Monitoring resumed. I'm watching."

        elif any(w in c for w in ["fine", "okay", "ok", "awake", "good"]):
            self.state["dismissed"] = True
            response = "Understood. Stay focused on the road."

        elif any(w in c for w in ["status", "check", "how am i"]):
            ear  = self.state.get("current_ear", 0.3)
            risk = self._get_risk(ear)
            dur  = self._get_duration()
            response = f"Fatigue risk is {risk}. Driving for {dur}."

        elif any(w in c for w in ["how long", "duration", "long"]):
            response = f"You have been driving for {self._get_duration()}."

        elif any(w in c for w in ["break", "rest", "tired", "pull"]):
            response = "Please pull over safely and rest for 15 minutes."

        elif any(w in c for w in ["breath", "breathing", "relax", "calm"]):
            threading.Thread(target=self._breathing_exercise, daemon=True).start()
            response = "Starting breathing exercise. Follow my guidance."

        elif any(w in c for w in ["water", "drink", "hydrat"]):
            response = "Drink water every 30 minutes on long drives."

        elif any(w in c for w in ["focus", "tip", "advice"]):
            response = "Keep the car cool, listen to upbeat music, stop every 2 hours."

        elif any(w in c for w in ["emergency", "danger", "crash", "help"]):
            response = "Emergency. Pull over immediately. Turn on hazard lights. Call for help."

        elif any(w in c for w in ["hello", "hi", "hey"]):
            response = f"Hello! Driving for {self._get_duration()}. How can I help?"

        elif any(w in c for w in ["command", "what can", "help"]):
            response = "Say Nova followed by: start, stop, pause, resume, status, how long, break, breathing, water, focus, or emergency."

        else:
            response = f"I heard {command}. Try saying Nova help for available commands."

        if response:
            self.speak(response)
    
    def _get_duration(self) -> str:
        start = self.state.get("session_start")
        if not start:
            return "an unknown duration"
        elapsed = int(time.time() - start)
        hours   = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        if hours > 0:
            return f"{hours} hour{'s' if hours>1 else ''} and {minutes} minutes"
        return f"{minutes} minute{'s' if minutes!=1 else ''}"

    def _get_risk(self, ear: float) -> str:
        if ear > 0.30:   return "Low"
        elif ear > 0.25: return "Moderate"
        elif ear > 0.20: return "High"
        else:            return "Critical"

    def _breathing_exercise(self):
        self.speak("Breathing exercise. Breathe in for 4 seconds.")
        time.sleep(5)
        self.speak("Hold for 4 seconds.")
        time.sleep(5)
        self.speak("Breathe out slowly.")
        time.sleep(5)
        self.speak("Well done. You should feel more alert now.")

    def stop(self):
        self.running = False