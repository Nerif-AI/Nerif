import os
from pathlib import Path
from typing import Optional, Union

import httpx

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=120.0)


class AudioModel:
    """Audio transcription via OpenAI-compatible /audio/transcriptions endpoint."""

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: str = None,
        base_url: str = None,
        language: Optional[str] = None,
        response_format: str = "json",
        counter=None,
    ):
        self.model = model
        self.api_key = api_key or _OPENAI_API_KEY
        self.base_url = (base_url or _OPENAI_API_BASE).rstrip("/")
        self.language = language
        self.response_format = response_format
        self.counter = counter

    def transcribe(
        self,
        file: Union[str, Path, tuple],
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Transcribe audio file.

        Args:
            file: Path to audio file, file-like object, or (filename, bytes) tuple.
            prompt: Optional prompt to guide transcription.
            language: ISO-639-1 language code (overrides instance default).
            response_format: Override instance default format.
            temperature: Sampling temperature (0-1).

        Returns:
            API response dict. For json format: {"text": "..."}.
        """
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        data = {"model": self.model}
        if prompt is not None:
            data["prompt"] = prompt
        lang = language or self.language
        if lang is not None:
            data["language"] = lang
        fmt = response_format or self.response_format
        if fmt is not None:
            data["response_format"] = fmt
        if temperature is not None:
            data["temperature"] = str(temperature)

        opened_file = None
        try:
            if isinstance(file, tuple):
                files = {"file": file}
            elif isinstance(file, (str, Path)):
                opened_file = open(file, "rb")
                files = {"file": (str(file), opened_file)}
            else:
                files = {"file": ("audio", file)}

            resp = httpx.post(url, headers=headers, files=files, data=data, timeout=_DEFAULT_TIMEOUT)
            resp.raise_for_status()
        finally:
            if opened_file is not None:
                opened_file.close()

        return resp.json()

    async def atranscribe(
        self,
        file: Union[str, Path, tuple],
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Async version of transcribe()."""
        url = f"{self.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        data = {"model": self.model}
        if prompt is not None:
            data["prompt"] = prompt
        lang = language or self.language
        if lang is not None:
            data["language"] = lang
        fmt = response_format or self.response_format
        if fmt is not None:
            data["response_format"] = fmt
        if temperature is not None:
            data["temperature"] = str(temperature)

        opened_file = None
        try:
            if isinstance(file, tuple):
                files = {"file": file}
            elif isinstance(file, (str, Path)):
                opened_file = open(file, "rb")
                files = {"file": (str(file), opened_file)}
            else:
                files = {"file": ("audio", file)}

            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, files=files, data=data)
                resp.raise_for_status()
        finally:
            if opened_file is not None:
                opened_file.close()

        return resp.json()

    def translate(self, file: Union[str, Path, tuple], prompt: Optional[str] = None) -> dict:
        """Translate audio to English text via /audio/translations endpoint."""
        url = f"{self.base_url}/audio/translations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model}
        if prompt is not None:
            data["prompt"] = prompt

        opened_file = None
        try:
            if isinstance(file, tuple):
                files = {"file": file}
            elif isinstance(file, (str, Path)):
                opened_file = open(file, "rb")
                files = {"file": (str(file), opened_file)}
            else:
                files = {"file": ("audio", file)}
            resp = httpx.post(url, headers=headers, files=files, data=data, timeout=_DEFAULT_TIMEOUT)
            resp.raise_for_status()
        finally:
            if opened_file is not None:
                opened_file.close()
        return resp.json()

    async def atranslate(self, file: Union[str, Path, tuple], prompt: Optional[str] = None) -> dict:
        """Async version of translate()."""
        url = f"{self.base_url}/audio/translations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model}
        if prompt is not None:
            data["prompt"] = prompt

        opened_file = None
        try:
            if isinstance(file, tuple):
                files = {"file": file}
            elif isinstance(file, (str, Path)):
                opened_file = open(file, "rb")
                files = {"file": (str(file), opened_file)}
            else:
                files = {"file": ("audio", file)}
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, files=files, data=data)
                resp.raise_for_status()
        finally:
            if opened_file is not None:
                opened_file.close()
        return resp.json()


class SpeechModel:
    """Text-to-speech via OpenAI-compatible /audio/speech endpoint."""

    VOICES = ("alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer")

    def __init__(
        self,
        model: str = "tts-1",
        api_key: str = None,
        base_url: str = None,
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
        counter=None,
    ):
        self.model = model
        self.api_key = api_key or _OPENAI_API_KEY
        self.base_url = (base_url or _OPENAI_API_BASE).rstrip("/")
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.counter = counter

    def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Generate speech audio from text. Returns raw audio bytes."""
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model if model is not None else self.model,
            "input": text,
            "voice": voice if voice is not None else self.voice,
            "response_format": response_format if response_format is not None else self.response_format,
            "speed": speed if speed is not None else self.speed,
        }
        resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    async def atext_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """Async version of text_to_speech()."""
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model if model is not None else self.model,
            "input": text,
            "voice": voice if voice is not None else self.voice,
            "response_format": response_format if response_format is not None else self.response_format,
            "speed": speed if speed is not None else self.speed,
        }
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
        return resp.content

    def text_to_speech_file(self, text: str, output_path: Union[str, Path], **kwargs) -> Path:
        """Generate speech and save to file. Returns the output path."""
        audio_bytes = self.text_to_speech(text, **kwargs)
        path = Path(output_path)
        path.write_bytes(audio_bytes)
        return path

    async def atext_to_speech_file(self, text: str, output_path: Union[str, Path], **kwargs) -> Path:
        """Async version of text_to_speech_file()."""
        audio_bytes = await self.atext_to_speech(text, **kwargs)
        path = Path(output_path)
        path.write_bytes(audio_bytes)
        return path
