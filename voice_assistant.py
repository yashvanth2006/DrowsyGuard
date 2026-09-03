import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import numpy as np
import logging
import config

logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self, command_queue: queue.Queue, state_getter):
        self.command_queue = command_queue
        self.state_getter = state_getter
        self.name = config.ASSISTANT_NAME
        
        self._running_event = threading.Event()
        self._listener_thread = None
        self._tts_thread_obj = None
        
        self.tts_queue = queue.Queue()
        self.last_spoken = {}
        
        self.recognizer = sr.Recognizer()
        self.command_recognizer = sr.Recognizer()
        
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.5
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.3
        
        self.command_recognizer.energy_threshold = 300
        self.command_recognizer.dynamic_energy_threshold = False
        self.command_recognizer.pause_threshold = 0.8
        
        self.vosk_model = None
        if config.VOICE_ENABLED:
            self._load_vosk()
            
    def _load_vosk(self):
        try:
            from vosk import Model
            import os
            if os.path.exists(config.VOSK_MODEL_PATH):
                self.vosk_model = Model(str(config.VOSK_MODEL_PATH))
                logger.debug("Vosk model loaded")
            else:
                logger.debug("Vosk model not found, using fallback")
        except Exception:
            logger.debug("Vosk fallback unavailable")

    def start(self):
        if not config.VOICE_ENABLED:
            self._dispatch_status("○ Disabled")
            return
            
        if self._running_event.is_set():
            return # Already running
            
        self._running_event.set()
        
        if config.TTS_ENABLED:
            self._tts_thread_obj = threading.Thread(target=self._tts_thread, daemon=True)
            self._tts_thread_obj.start()
            
        self._listener_thread = threading.Thread(target=self._wake_word_pipeline, daemon=True)
        self._listener_thread.start()
        
    def stop(self):
        self._running_event.clear()
        if self._listener_thread and self._listener_thread is not threading.current_thread():
            # Don't join indefinitely to avoid blocking
            self._listener_thread.join(timeout=1.0)
            self._listener_thread = None
        if self._tts_thread_obj and self._tts_thread_obj is not threading.current_thread():
            self._tts_thread_obj.join(timeout=1.0)
            self._tts_thread_obj = None
        self._dispatch_status("○ Disabled")

    def is_running(self):
        return self._running_event.is_set()
        
    def _tts_thread(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_RATE)
            engine.setProperty("volume", config.TTS_VOLUME)
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            return
            
        while self._running_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.5)
                self._dispatch_status("● Speaking")
                now = time.time()
                if text in self.last_spoken:
                    if now - self.last_spoken[text] < 3:
                        continue
                self.last_spoken[text] = now
                engine.say(text)
                engine.runAndWait()
                self._dispatch_status("● Listening")
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"TTS error: {e}")

    def speak(self, text: str):
        if config.TTS_ENABLED:
            self.tts_queue.put(text)
            
    def _play_activation_beep(self):
        if not config.AUDIO_ALERT_ENABLED:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            sample_rate = config.AUDIO_SAMPLE_RATE
            t = np.linspace(0, 0.15, int(sample_rate * 0.15))
            wave = (np.sin(2 * np.pi * 880 * t) * 16383).astype(np.int16)
            stereo = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound(stereo)
            sound.play()
        except Exception:
            pass

    def _is_wake_word(self, text: str) -> bool:
        text = text.lower().strip()
        return any(w in text for w in config.WAKE_WORDS)

    def _wake_word_pipeline(self):
        self._dispatch_status("● Listening")
        while self._running_event.is_set():
            try:
                with sr.Microphone() as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=config.WAKE_WORD_TIMEOUT,
                        phrase_time_limit=config.WAKE_WORD_PHRASE_TIME_LIMIT
                    )
                
                wake_detected = False
                
                if self.vosk_model:
                    wake_detected = self._check_vosk(audio)
                    
                if not wake_detected:
                    try:
                        text = self.recognizer.recognize_google(audio, show_all=False).lower()
                        wake_detected = self._is_wake_word(text)
                    except (sr.UnknownValueError, sr.RequestError):
                        pass

                if wake_detected:
                    self._dispatch_status("◐ Processing (Wake Word)")
                    self._play_activation_beep()
                    
                    try:
                        with sr.Microphone() as source:
                            command_audio = self.command_recognizer.listen(
                                source,
                                timeout=config.COMMAND_TIMEOUT,
                                phrase_time_limit=config.COMMAND_PHRASE_TIME_LIMIT
                            )
                        
                        command = self.command_recognizer.recognize_google(command_audio).lower()
                        self._dispatch_status(f"◐ Processing: {command}")
                        self._process_command(command)
                        
                    except sr.UnknownValueError:
                        self.speak("I didn't catch that. Please try again.")
                        self._dispatch_status("● Listening")
                    except sr.WaitTimeoutError:
                        self._dispatch_status("● Listening")
                    except sr.RequestError:
                        self._dispatch_status("● Listening")
                    
                    time.sleep(0.5)

            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                logger.error(f"Voice pipeline error: {e}")
                self._dispatch_status("○ Disabled (Error)")
                self.stop()
                break
                
    def _check_vosk(self, audio) -> bool:
        try:
            from vosk import KaldiRecognizer
            import json
            rec = KaldiRecognizer(self.vosk_model, 16000)
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            rec.AcceptWaveform(raw)
            result = json.loads(rec.Result())
            text = result.get("text", "").lower()
            return self._is_wake_word(text)
        except Exception:
            return False

    def _dispatch_status(self, status: str):
        self.command_queue.put({"action": "NOVA_STATUS", "status": status})

    def _process_command(self, command: str):
        c = command.lower()
        response = None
        
        if any(w in c for w in ["start ", "start", "begin ", "begin", "go "]):
            self.command_queue.put({"action": "START"})
            response = "Monitoring started. Stay alert and drive safely."

        elif any(w in c for w in ["stop", "end", "finish"]):
            self.command_queue.put({"action": "STOP"})
            response = "Monitoring stopped. Drive safely."
            
        elif any(w in c for w in ["status", "check", "how am i"]):
            state = self.state_getter()
            if state:
                monitoring = "active" if state.get("monitoring") else "inactive"
                risk = state.get("risk_level", "Unknown")
                alerts = state.get("alert_count", 0)
                dur = state.get("session_duration", "unknown time")
                response = f"Monitoring is {monitoring}. Risk level is {risk}. You have {alerts} alerts. Driving for {dur}."
            else:
                response = "I cannot determine your status right now."

        elif any(w in c for w in ["how long", "duration", "long"]):
            state = self.state_getter()
            if state:
                dur = state.get("session_duration", "an unknown duration")
                response = f"You have been driving for {dur}."

        elif any(w in c for w in ["break", "rest", "tired", "pull"]):
            response = "Please pull over safely and rest for 15 minutes."

        elif any(w in c for w in ["water", "drink", "hydrat"]):
            response = "Drink water every 30 minutes on long drives."

        elif any(w in c for w in ["focus", "tip", "advice"]):
            response = "Keep the car cool, listen to upbeat music, stop every 2 hours."

        elif any(w in c for w in ["emergency", "danger", "crash", "help"]):
            response = "Emergency. Pull over immediately. Turn on hazard lights. Call for help."
            
        elif any(w in c for w in ["hello", "hi", "hey"]):
            response = "Hello! I am Nova, your voice assistant. How can I help?"
            
        elif any(w in c for w in ["command", "what can", "help"]):
            response = "Say Nova followed by: start monitoring, stop monitoring, status, duration, break, or emergency."

        else:
            response = f"I heard {command}. Try saying Nova help for available commands."

        if response:
            self.speak(response)