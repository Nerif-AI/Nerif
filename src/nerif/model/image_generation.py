import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx

from ..utils.constants import DEFAULT_TIMEOUT as _DEFAULT_TIMEOUT

_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
_OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
_GOOGLE_API_BASE = os.environ.get("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com")


@dataclass
class GeneratedImage:
    """Normalized image generation output."""

    b64_json: Optional[str] = None
    url: Optional[str] = None
    mime_type: Optional[str] = None
    revised_prompt: Optional[str] = None

    def as_bytes(self) -> bytes:
        if not self.b64_json:
            raise ValueError("This image does not contain inline base64 data")
        return base64.b64decode(self.b64_json)

    def save(self, path: Union[str, Path]) -> Path:
        output_path = Path(path)
        output_path.write_bytes(self.as_bytes())
        return output_path


@dataclass
class ImageGenerationResult:
    """Collection of generated images plus provider metadata."""

    created: Optional[int] = None
    data: List[GeneratedImage] = field(default_factory=list)
    text: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


def _read_image_as_base64(image: Union[str, Path, bytes]) -> str:
    if isinstance(image, bytes):
        raw = image
    else:
        raw = Path(image).read_bytes()
    return base64.b64encode(raw).decode("utf-8")


class ImageGenerationModel:
    """OpenAI-compatible image generation wrapper."""

    def __init__(
        self,
        model: str = "gpt-image-1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or _OPENAI_API_KEY
        self.base_url = (base_url or _OPENAI_API_BASE).rstrip("/")

    def generate(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: Optional[str] = None,
        background: Optional[str] = None,
        moderation: Optional[str] = None,
        output_format: Optional[str] = None,
        output_compression: Optional[int] = None,
        user: Optional[str] = None,
    ) -> ImageGenerationResult:
        url = f"{self.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if quality is not None:
            body["quality"] = quality
        if background is not None:
            body["background"] = background
        if moderation is not None:
            body["moderation"] = moderation
        if output_format is not None:
            body["output_format"] = output_format
        if output_compression is not None:
            body["output_compression"] = output_compression
        if user is not None:
            body["user"] = user

        resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        images = [
            GeneratedImage(
                b64_json=item.get("b64_json"),
                url=item.get("url"),
                revised_prompt=item.get("revised_prompt"),
            )
            for item in payload.get("data", [])
        ]
        return ImageGenerationResult(created=payload.get("created"), data=images, raw_response=payload)

    async def agenerate(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        quality: Optional[str] = None,
        background: Optional[str] = None,
        moderation: Optional[str] = None,
        output_format: Optional[str] = None,
        output_compression: Optional[int] = None,
        user: Optional[str] = None,
    ) -> ImageGenerationResult:
        """Async version of generate."""
        url = f"{self.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if quality is not None:
            body["quality"] = quality
        if background is not None:
            body["background"] = background
        if moderation is not None:
            body["moderation"] = moderation
        if output_format is not None:
            body["output_format"] = output_format
        if output_compression is not None:
            body["output_compression"] = output_compression
        if user is not None:
            body["user"] = user

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        images = [
            GeneratedImage(
                b64_json=item.get("b64_json"),
                url=item.get("url"),
                revised_prompt=item.get("revised_prompt"),
            )
            for item in payload.get("data", [])
        ]
        return ImageGenerationResult(created=payload.get("created"), data=images, raw_response=payload)


class NanoBananaModel:
    """Gemini image generation and image editing wrapper."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-image-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or _GOOGLE_API_KEY
        self.base_url = (base_url or _GOOGLE_API_BASE).rstrip("/")

    def generate(
        self,
        prompt: str,
        images: Optional[Sequence[Union[str, Path, bytes]]] = None,
        mime_type: str = "image/png",
    ) -> ImageGenerationResult:
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        parts: List[Dict[str, Any]] = [{"text": prompt}]

        for image in images or []:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": _read_image_as_base64(image),
                    }
                }
            )

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }
        headers = {"Content-Type": "application/json"}

        resp = httpx.post(url, json=body, headers=headers, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        text_chunks: List[str] = []
        result_images: List[GeneratedImage] = []
        for candidate in payload.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text_chunks.append(part["text"])
                inline_data = part.get("inlineData")
                if inline_data:
                    result_images.append(
                        GeneratedImage(
                            b64_json=inline_data.get("data"),
                            mime_type=inline_data.get("mimeType"),
                        )
                    )

        return ImageGenerationResult(
            data=result_images,
            text="\n".join(text_chunks) if text_chunks else None,
            raw_response=payload,
        )

    async def agenerate(
        self,
        prompt: str,
        images: Optional[Sequence[Union[str, Path, bytes]]] = None,
        mime_type: str = "image/png",
    ) -> ImageGenerationResult:
        """Async version of generate."""
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        parts: List[Dict[str, Any]] = [{"text": prompt}]

        for image in images or []:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": _read_image_as_base64(image),
                    }
                }
            )

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
            },
        }
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        text_chunks: List[str] = []
        result_images: List[GeneratedImage] = []
        for candidate in payload.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    text_chunks.append(part["text"])
                inline_data = part.get("inlineData")
                if inline_data:
                    result_images.append(
                        GeneratedImage(
                            b64_json=inline_data.get("data"),
                            mime_type=inline_data.get("mimeType"),
                        )
                    )

        return ImageGenerationResult(
            data=result_images,
            text="\n".join(text_chunks) if text_chunks else None,
            raw_response=payload,
        )
