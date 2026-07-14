import json
import os

SETTINGS_FILE = "assistant_settings.json"

DEFAULT_SETTINGS = {
    "assistant_name": None,
    "voice_speed": "normal",
    "alert_volume": 1.0,
    "auto_start": False
}

def save_settings(data: dict):
    existing = load_settings()
    existing.update(data)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()

def get_assistant_name() -> str | None:
    return load_settings().get("assistant_name")

def set_assistant_name(name: str):
    save_settings({"assistant_name": name.strip().lower()})