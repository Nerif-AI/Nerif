from typing import Any, Optional

from .model import MultiModalMessage, SimpleChatModel


class AudioInferenceModel:
    """A wrapper around SimpleChatModel for audio multimodal reasoning."""

    def __init__(
        self,
        model: str,
        default_prompt: str = "You are a helpful assistant that can understand and reason over audio inputs.",
        temperature: float = 0.0,
        counter: Optional[Any] = None,
        max_tokens: Optional[int] = None,
    ):
        self._chat_model = SimpleChatModel(
            model=model,
            default_prompt=default_prompt,
            temperature=temperature,
            counter=counter,
            max_tokens=max_tokens,
        )

    def analyze_url(
        self, audio_url: str, prompt: str = "Describe this audio.", max_tokens: Optional[int] = None
    ) -> Any:
        msg = MultiModalMessage().add_audio_url(audio_url).add_text(prompt)
        return self._chat_model.chat(msg, max_tokens=max_tokens)

    def analyze_path(
        self,
        audio_path: str,
        prompt: str = "Transcribe and analyze this audio.",
        format: str = "wav",
        max_tokens: Optional[int] = None,
    ) -> Any:
        msg = MultiModalMessage().add_audio_path(audio_path, format=format).add_text(prompt)
        return self._chat_model.chat(msg, max_tokens=max_tokens)

    def analyze_base64(
        self,
        audio_base64: str,
        prompt: str = "Transcribe and analyze this audio.",
        format: str = "wav",
        max_tokens: Optional[int] = None,
    ) -> Any:
        msg = MultiModalMessage().add_audio_base64(audio_base64, format=format).add_text(prompt)
        return self._chat_model.chat(msg, max_tokens=max_tokens)

    async def aanalyze_url(
        self, audio_url: str, prompt: str = "Describe this audio.", max_tokens: Optional[int] = None
    ) -> Any:
        """Async version of analyze_url."""
        msg = MultiModalMessage().add_audio_url(audio_url).add_text(prompt)
        return await self._chat_model.achat(msg, max_tokens=max_tokens)

    async def aanalyze_path(
        self,
        audio_path: str,
        prompt: str = "Transcribe and analyze this audio.",
        format: str = "wav",
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Async version of analyze_path."""
        msg = MultiModalMessage().add_audio_path(audio_path, format=format).add_text(prompt)
        return await self._chat_model.achat(msg, max_tokens=max_tokens)

    async def aanalyze_base64(
        self,
        audio_base64: str,
        prompt: str = "Transcribe and analyze this audio.",
        format: str = "wav",
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Async version of analyze_base64."""
        msg = MultiModalMessage().add_audio_base64(audio_base64, format=format).add_text(prompt)
        return await self._chat_model.achat(msg, max_tokens=max_tokens)

    def reset(self) -> None:
        self._chat_model.reset()
