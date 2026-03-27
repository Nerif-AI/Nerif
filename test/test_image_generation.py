"""Tests for ImageGenerationModel, NanoBananaModel, GeneratedImage, and ImageGenerationResult."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nerif.img_gen import GeneratedImage, ImageGenerationModel, ImageGenerationResult, NanoBananaModel


# ===========================================================================
# GeneratedImage dataclass
# ===========================================================================
class TestGeneratedImage:
    def test_default_fields_are_none(self):
        img = GeneratedImage()
        assert img.b64_json is None
        assert img.url is None
        assert img.mime_type is None
        assert img.revised_prompt is None

    def test_as_bytes_decodes_base64(self):
        raw = b"hello world"
        img = GeneratedImage(b64_json=base64.b64encode(raw).decode())
        assert img.as_bytes() == raw

    def test_as_bytes_raises_when_no_b64(self):
        img = GeneratedImage(url="https://example.com/img.png")
        with pytest.raises(ValueError, match="does not contain inline base64"):
            img.as_bytes()

    def test_save_writes_file(self, tmp_path):
        raw = b"image bytes"
        img = GeneratedImage(b64_json=base64.b64encode(raw).decode())
        out = img.save(tmp_path / "out.png")
        assert isinstance(out, Path)
        assert out.read_bytes() == raw

    def test_save_returns_path_object(self, tmp_path):
        img = GeneratedImage(b64_json=base64.b64encode(b"x").decode())
        result = img.save(str(tmp_path / "f.bin"))
        assert isinstance(result, Path)


# ===========================================================================
# ImageGenerationResult dataclass
# ===========================================================================
class TestImageGenerationResult:
    def test_default_fields(self):
        r = ImageGenerationResult()
        assert r.created is None
        assert r.data == []
        assert r.text is None
        assert r.raw_response is None

    def test_data_is_independent_per_instance(self):
        r1 = ImageGenerationResult()
        r2 = ImageGenerationResult()
        r1.data.append(GeneratedImage(b64_json="abc"))
        assert len(r2.data) == 0


# ===========================================================================
# ImageGenerationModel (OpenAI-compatible)
# ===========================================================================
class TestImageGenerationModel:
    def test_default_model_name(self):
        m = ImageGenerationModel(api_key="k")
        assert m.model == "gpt-image-1"

    def test_custom_model_and_base_url(self):
        m = ImageGenerationModel(model="dall-e-3", api_key="k", base_url="https://my.api/v1/")
        assert m.model == "dall-e-3"
        assert m.base_url == "https://my.api/v1"  # trailing slash stripped

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_sends_correct_request(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "created": 12345,
            "data": [{"b64_json": "YWJj", "revised_prompt": "revised"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="test-key", base_url="https://api.example.com/v1")
        result = model.generate("draw a cat", size="512x512")

        assert result.created == 12345
        assert len(result.data) == 1
        assert result.data[0].b64_json == "YWJj"
        assert result.data[0].revised_prompt == "revised"

        call_kwargs = mock_post.call_args
        url = call_kwargs[0][0]
        assert url == "https://api.example.com/v1/images/generations"
        body = call_kwargs.kwargs["json"]
        assert body["model"] == "gpt-image-1"
        assert body["prompt"] == "draw a cat"
        assert body["size"] == "512x512"
        assert body["n"] == 1

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_includes_optional_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"created": 1, "data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="k", base_url="https://api.example.com/v1")
        model.generate(
            "prompt",
            n=2,
            quality="hd",
            background="transparent",
            moderation="low",
            output_format="png",
            output_compression=90,
            user="user-123",
        )

        body = mock_post.call_args.kwargs["json"]
        assert body["n"] == 2
        assert body["quality"] == "hd"
        assert body["background"] == "transparent"
        assert body["moderation"] == "low"
        assert body["output_format"] == "png"
        assert body["output_compression"] == 90
        assert body["user"] == "user-123"

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_omits_none_optional_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"created": 1, "data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="k", base_url="https://api.example.com/v1")
        model.generate("prompt")

        body = mock_post.call_args.kwargs["json"]
        assert "quality" not in body
        assert "background" not in body
        assert "moderation" not in body
        assert "output_format" not in body
        assert "output_compression" not in body
        assert "user" not in body

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_sets_auth_header(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="secret-key", base_url="https://api.example.com/v1")
        model.generate("prompt")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-key"
        assert headers["Content-Type"] == "application/json"

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_multiple_images(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "created": 1,
            "data": [
                {"b64_json": "aW1n", "url": None, "revised_prompt": "p1"},
                {"b64_json": "aW1n", "url": "https://img.url/2.png", "revised_prompt": "p2"},
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="k", base_url="https://api.example.com/v1")
        result = model.generate("prompt", n=2)

        assert len(result.data) == 2
        assert result.data[1].url == "https://img.url/2.png"

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_preserves_raw_response(self, mock_post):
        raw = {"created": 99, "data": [], "extra_field": "value"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="k", base_url="https://api.example.com/v1")
        result = model.generate("prompt")
        assert result.raw_response == raw

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_empty_data(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"created": 1}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = ImageGenerationModel(api_key="k", base_url="https://api.example.com/v1")
        result = model.generate("prompt")
        assert result.data == []


# ===========================================================================
# NanoBananaModel (Gemini)
# ===========================================================================
class TestNanoBananaModel:
    def test_default_model_name(self):
        m = NanoBananaModel(api_key="k")
        assert m.model == "gemini-2.5-flash-image-preview"

    def test_custom_model_and_base_url(self):
        m = NanoBananaModel(model="gemini-2.0-flash", api_key="k", base_url="https://custom.api/")
        assert m.model == "gemini-2.0-flash"
        assert m.base_url == "https://custom.api"

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_text_only_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "done"}, {"inlineData": {"mimeType": "image/png", "data": "YWJj"}}]}}
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        result = model.generate("draw a banana astronaut")

        assert result.text == "done"
        assert len(result.data) == 1
        assert result.data[0].mime_type == "image/png"
        assert result.data[0].b64_json == "YWJj"

        call_kwargs = mock_post.call_args
        url = call_kwargs[0][0]
        assert "gemini-2.5-flash-image-preview:generateContent" in url
        assert "key=gk" in url
        body = call_kwargs.kwargs["json"]
        parts = body["contents"][0]["parts"]
        assert parts[0] == {"text": "draw a banana astronaut"}
        assert body["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_with_image_input_bytes(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "edited"}]}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        input_bytes = b"fake png bytes"
        result = model.generate("edit this image", images=[input_bytes], mime_type="image/png")

        assert result.text == "edited"
        body = mock_post.call_args.kwargs["json"]
        parts = body["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "edit this image"}
        assert parts[1]["inlineData"]["mimeType"] == "image/png"
        expected_b64 = base64.b64encode(input_bytes).decode()
        assert parts[1]["inlineData"]["data"] == expected_b64

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_with_image_input_path(self, mock_post, tmp_path):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        img_path = tmp_path / "input.png"
        img_path.write_bytes(b"img data")

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        model.generate("describe", images=[str(img_path)])

        body = mock_post.call_args.kwargs["json"]
        parts = body["contents"][0]["parts"]
        assert len(parts) == 2
        expected_b64 = base64.b64encode(b"img data").decode()
        assert parts[1]["inlineData"]["data"] == expected_b64

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_with_multiple_image_inputs(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "merged"}]}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        model.generate("merge", images=[b"img1", b"img2"])

        body = mock_post.call_args.kwargs["json"]
        parts = body["contents"][0]["parts"]
        assert len(parts) == 3  # text + 2 images

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_no_images_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": "text only"}]}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        result = model.generate("just chat")

        assert result.text == "text only"
        assert result.data == []

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_no_text_in_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": "abc"}}]}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        result = model.generate("image only")

        assert result.text is None
        assert len(result.data) == 1

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_empty_candidates(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"candidates": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        result = model.generate("prompt")

        assert result.text is None
        assert result.data == []

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_preserves_raw_response(self, mock_post):
        raw = {"candidates": [], "usageMetadata": {"promptTokenCount": 10}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = raw
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        result = model.generate("prompt")
        assert result.raw_response == raw

    @patch("nerif.model.image_generation.httpx.post")
    def test_generate_multiple_candidates(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "part1"}]}},
                {
                    "content": {
                        "parts": [{"text": "part2"}, {"inlineData": {"mimeType": "image/png", "data": "ZGF0YQ=="}}]
                    }
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        model = NanoBananaModel(api_key="gk", base_url="https://generativelanguage.googleapis.com")
        result = model.generate("prompt")

        assert result.text == "part1\npart2"
        assert len(result.data) == 1
