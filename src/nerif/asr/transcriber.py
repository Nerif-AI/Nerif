"""High-level speech-to-text interface."""

from typing import Optional

from ..model.audio_model import AudioModel


class Transcriber:
    """High-level speech-to-text interface.

    Wraps AudioModel with a simplified API that returns plain text.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: str = None,
        base_url: str = None,
        language: Optional[str] = None,
        counter=None,
    ):
        self._audio_model = AudioModel(
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            counter=counter,
        )

    def transcribe(self, file, **kwargs) -> str:
        """Transcribe and return plain text."""
        result = self._audio_model.transcribe(file, **kwargs)
        return result.get("text", "")

    async def atranscribe(self, file, **kwargs) -> str:
        """Async version."""
        result = await self._audio_model.atranscribe(file, **kwargs)
        return result.get("text", "")

    def translate(self, file, **kwargs) -> str:
        """Translate audio to English and return plain text."""
        result = self._audio_model.translate(file, **kwargs)
        return result.get("text", "")

    async def atranslate(self, file, **kwargs) -> str:
        """Async version."""
        result = await self._audio_model.atranslate(file, **kwargs)
        return result.get("text", "")
