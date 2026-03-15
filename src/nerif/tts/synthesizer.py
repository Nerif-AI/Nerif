"""High-level text-to-speech interface."""

from pathlib import Path
from typing import Union

from ..model.audio_model import SpeechModel


class Synthesizer:
    """High-level text-to-speech interface.

    Provides convenient methods for speech generation with
    sensible defaults and file output.
    """

    def __init__(
        self,
        model: str = "tts-1",
        voice: str = "alloy",
        api_key: str = None,
        base_url: str = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        counter=None,
    ):
        self._speech_model = SpeechModel(
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
            response_format=response_format,
            speed=speed,
            counter=counter,
        )

    @property
    def available_voices(self) -> tuple:
        return SpeechModel.VOICES

    def speak(self, text: str, **kwargs) -> bytes:
        """Generate speech audio bytes."""
        return self._speech_model.text_to_speech(text, **kwargs)

    async def aspeak(self, text: str, **kwargs) -> bytes:
        """Async version."""
        return await self._speech_model.atext_to_speech(text, **kwargs)

    def speak_to_file(self, text: str, output_path: Union[str, Path], **kwargs) -> Path:
        """Generate speech and save to file."""
        return self._speech_model.text_to_speech_file(text, output_path, **kwargs)

    async def aspeak_to_file(self, text: str, output_path: Union[str, Path], **kwargs) -> Path:
        """Async version."""
        return await self._speech_model.atext_to_speech_file(text, output_path, **kwargs)
