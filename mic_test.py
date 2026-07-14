import argparse
import speech_recognition as sr


def main(device_index: int | None, duration: int):
    r = sr.Recognizer()

    try:
        names = sr.Microphone.list_microphone_names()
        print("Available microphones:")
        for i, n in enumerate(names):
            print(f"  {i}: {n}")
    except Exception as e:
        print(f"Could not list microphones: {e}")

    ctx = None
    try:
        if device_index is not None:
            print(f"Opening microphone index {device_index}")
            ctx = sr.Microphone(device_index=device_index)
        else:
            print("Opening default microphone")
            ctx = sr.Microphone()

        with ctx as source:
            print("Adjusting for ambient noise (2s)...")
            r.dynamic_energy_threshold = True
            r.adjust_for_ambient_noise(source, duration=2)
            print(f"Energy threshold after adjust: {r.energy_threshold}")
            print(f"Recording for {duration} seconds... Speak now.")
            audio = r.record(source, duration=duration)

        wav = audio.get_wav_data()
        with open("mic_test.wav", "wb") as f:
            f.write(wav)
        print("Saved recorded audio to mic_test.wav")

        try:
            print("Attempting Google speech recognition...")
            text = r.recognize_google(audio)
            print("Transcription:", text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Microphone test failed: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Microphone test recorder and recognizer")
    p.add_argument("--device", type=int, default=None, help="Microphone device index (optional)")
    p.add_argument("--duration", type=int, default=5, help="Seconds to record")
    args = p.parse_args()
    main(args.device, args.duration)
