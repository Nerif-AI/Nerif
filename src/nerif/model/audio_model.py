import os
from pathlib import Path

import httpx

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=120.0)


class AudioModel:
    """
    A simple agent for audio tasks. (audio transcription, speech to text)
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or _OPENAI_API_KEY
        self.base_url = (base_url or _OPENAI_API_BASE).rstrip("/")

    def transcribe(self, file: Path):
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with open(file, "rb") as audio_file:
            files = {"file": (str(file), audio_file)}
            data = {"model": "whisper-1"}
            resp = httpx.post(url, headers=headers, files=files, data=data, timeout=_DEFAULT_TIMEOUT)
            resp.raise_for_status()
        return resp.json()


class SpeechModel:
    """
    A simple agent for speech tasks. (speech model, text to speech)
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or _OPENAI_API_KEY
        self.base_url = (base_url or _OPENAI_API_BASE).rstrip("/")

    def text_to_speech(self, text: str, voice: str = "alloy"):
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
        }
        resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.content
