"""Comprehensive pydantic integration tests.

Covers:
- Structured output via response_model= (sync and async)
- Complex model types (nested, enum, literal, union, constrained fields)
- LLM returns malformed JSON — graceful error handling
- LLM returns extra fields — pydantic strips/ignores them
- LLM returns JSON wrapped in markdown code blocks
- Normal chat (no response_model) is unaffected by pydantic being installed
- response_model + append=True preserves history correctly
- response_model + fallback interaction
- FormatVerifierPydantic edge cases
"""

import asyncio
import json
from enum import Enum
from typing import Dict, List, Literal, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, field_validator

from nerif.model.model import SimpleChatModel
from nerif.utils.format import FormatVerifierPydantic

# ---------------------------------------------------------------------------
# Test models — ranging from simple to complex
# ---------------------------------------------------------------------------


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class SimpleItem(BaseModel):
    name: str
    value: int


class ItemWithDefaults(BaseModel):
    name: str
    count: int = 0
    active: bool = True


class ItemWithConstraints(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=1.0)


class ItemWithValidator(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("must contain @")
        return v


class NestedAddress(BaseModel):
    street: str
    city: str
    zip_code: Optional[str] = None


class PersonWithAddress(BaseModel):
    name: str
    age: int
    address: NestedAddress
    hobbies: List[str] = []


class ItemWithEnum(BaseModel):
    name: str
    color: Color


class ItemWithLiteral(BaseModel):
    name: str
    status: Literal["active", "inactive", "pending"]


class ItemWithDict(BaseModel):
    name: str
    metadata: Dict[str, str] = {}


class ItemWithUnion(BaseModel):
    value: Union[int, str]


# ---------------------------------------------------------------------------
# Helper to build mock LLM responses
# ---------------------------------------------------------------------------


def _mock_response(content: str):
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Basic: response_model returns pydantic instance
# ---------------------------------------------------------------------------


class TestResponseModelBasic:
    @patch("nerif.model.model.get_model_response")
    def test_simple_model(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "Widget", "value": 42}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("describe widget", response_model=SimpleItem)
        assert isinstance(result, SimpleItem)
        assert result.name == "Widget"
        assert result.value == 42

    @patch("nerif.model.model.get_model_response")
    def test_nested_model(self, mock):
        data = {
            "name": "Alice",
            "age": 30,
            "address": {"street": "123 Main", "city": "SF", "zip_code": "94102"},
            "hobbies": ["reading", "hiking"],
        }
        mock.return_value = _mock_response(json.dumps(data))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("tell me about Alice", response_model=PersonWithAddress)
        assert isinstance(result, PersonWithAddress)
        assert isinstance(result.address, NestedAddress)
        assert result.address.zip_code == "94102"
        assert result.hobbies == ["reading", "hiking"]

    @patch("nerif.model.model.get_model_response")
    def test_enum_field(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "Sky", "color": "blue"}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("color?", response_model=ItemWithEnum)
        assert result.color == Color.BLUE

    @patch("nerif.model.model.get_model_response")
    def test_literal_field(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "Task", "status": "active"}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("status?", response_model=ItemWithLiteral)
        assert result.status == "active"

    @patch("nerif.model.model.get_model_response")
    def test_dict_field(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "X", "metadata": {"key": "val"}}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("meta?", response_model=ItemWithDict)
        assert result.metadata == {"key": "val"}

    @patch("nerif.model.model.get_model_response")
    def test_union_field_int(self, mock):
        mock.return_value = _mock_response(json.dumps({"value": 42}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("val?", response_model=ItemWithUnion)
        assert result.value == 42

    @patch("nerif.model.model.get_model_response")
    def test_union_field_str(self, mock):
        mock.return_value = _mock_response(json.dumps({"value": "hello"}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("val?", response_model=ItemWithUnion)
        assert result.value == "hello"


# ---------------------------------------------------------------------------
# Defaults and optional fields
# ---------------------------------------------------------------------------


class TestResponseModelDefaults:
    @patch("nerif.model.model.get_model_response")
    def test_default_fields_filled(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "X"}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("item?", response_model=ItemWithDefaults)
        assert result.count == 0
        assert result.active is True

    @patch("nerif.model.model.get_model_response")
    def test_optional_nested_absent(self, mock):
        data = {"name": "Bob", "age": 25, "address": {"street": "1st", "city": "LA"}}
        mock.return_value = _mock_response(json.dumps(data))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("person?", response_model=PersonWithAddress)
        assert result.address.zip_code is None


# ---------------------------------------------------------------------------
# Constrained fields and validators
# ---------------------------------------------------------------------------


class TestResponseModelValidation:
    @patch("nerif.model.model.get_model_response")
    def test_constrained_valid(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "OK", "score": 0.85}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("score?", response_model=ItemWithConstraints)
        assert result.score == 0.85

    @patch("nerif.model.model.get_model_response")
    def test_custom_validator_valid(self, mock):
        mock.return_value = _mock_response(json.dumps({"email": "test@example.com"}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("email?", response_model=ItemWithValidator)
        assert result.email == "test@example.com"


# ---------------------------------------------------------------------------
# LLM returns malformed output — error handling
# ---------------------------------------------------------------------------


class TestResponseModelErrors:
    @patch("nerif.model.model.get_model_response")
    def test_invalid_json_raises(self, mock):
        mock.return_value = _mock_response("This is not JSON at all.")
        model = SimpleChatModel(model="gpt-4o")
        with pytest.raises(Exception):
            model.chat("something", response_model=SimpleItem)

    @patch("nerif.model.model.get_model_response")
    def test_missing_required_field_raises(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "Widget"}))  # missing 'value'
        model = SimpleChatModel(model="gpt-4o")
        with pytest.raises(Exception):
            model.chat("item?", response_model=SimpleItem)

    @patch("nerif.model.model.get_model_response")
    def test_wrong_type_raises(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "X", "value": "not_int"}))
        model = SimpleChatModel(model="gpt-4o")
        with pytest.raises(Exception):
            model.chat("item?", response_model=SimpleItem)

    @patch("nerif.model.model.get_model_response")
    def test_invalid_enum_raises(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "X", "color": "purple"}))
        model = SimpleChatModel(model="gpt-4o")
        with pytest.raises(Exception):
            model.chat("color?", response_model=ItemWithEnum)

    @patch("nerif.model.model.get_model_response")
    def test_constrained_out_of_range_raises(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "X", "score": 1.5}))
        model = SimpleChatModel(model="gpt-4o")
        with pytest.raises(Exception):
            model.chat("score?", response_model=ItemWithConstraints)

    @patch("nerif.model.model.get_model_response")
    def test_validator_failure_raises(self, mock):
        mock.return_value = _mock_response(json.dumps({"email": "not-an-email"}))
        model = SimpleChatModel(model="gpt-4o")
        with pytest.raises(Exception):
            model.chat("email?", response_model=ItemWithValidator)


# ---------------------------------------------------------------------------
# LLM returns extra fields — pydantic should ignore them
# ---------------------------------------------------------------------------


class TestResponseModelExtraFields:
    @patch("nerif.model.model.get_model_response")
    def test_extra_fields_ignored(self, mock):
        data = {"name": "Widget", "value": 42, "extra_field": "should be ignored", "another": 99}
        mock.return_value = _mock_response(json.dumps(data))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("item?", response_model=SimpleItem)
        assert isinstance(result, SimpleItem)
        assert result.name == "Widget"
        assert not hasattr(result, "extra_field")


# ---------------------------------------------------------------------------
# JSON wrapped in markdown code block (common LLM behavior)
# ---------------------------------------------------------------------------


class TestResponseModelMarkdownWrapped:
    @patch("nerif.model.model.get_model_response")
    def test_markdown_json_block(self, mock):
        data = {"name": "Widget", "value": 42}
        wrapped = "```json\n" + json.dumps(data) + "\n```"
        mock.return_value = _mock_response(wrapped)
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("item?", response_model=SimpleItem)
        assert isinstance(result, SimpleItem)
        assert result.value == 42

    @patch("nerif.model.model.get_model_response")
    def test_json_with_surrounding_text(self, mock):
        data = {"name": "Widget", "value": 42}
        surrounded = "Here is the result: " + json.dumps(data) + " Hope that helps!"
        mock.return_value = _mock_response(surrounded)
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("item?", response_model=SimpleItem)
        assert isinstance(result, SimpleItem)


# ---------------------------------------------------------------------------
# Normal chat without response_model — pydantic must not interfere
# ---------------------------------------------------------------------------


class TestNormalChatUnaffected:
    @patch("nerif.model.model.get_model_response")
    def test_plain_string_response(self, mock):
        mock.return_value = _mock_response("Hello, I'm a helpful assistant!")
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("Hi")
        assert isinstance(result, str)
        assert result == "Hello, I'm a helpful assistant!"

    @patch("nerif.model.model.get_model_response")
    def test_json_string_without_model_stays_string(self, mock):
        """Even if LLM returns valid JSON, without response_model it should stay as str."""
        mock.return_value = _mock_response('{"name": "test", "value": 1}')
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("return json")
        assert isinstance(result, str)
        assert result == '{"name": "test", "value": 1}'

    @patch("nerif.model.model.get_model_response")
    def test_response_model_none_explicitly(self, mock):
        mock.return_value = _mock_response("Just text")
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("hello", response_model=None)
        assert isinstance(result, str)

    @patch("nerif.model.model.get_model_response")
    def test_empty_response(self, mock):
        mock.return_value = _mock_response("")
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("empty?")
        assert isinstance(result, str)
        assert result == ""

    @patch("nerif.model.model.get_model_response")
    def test_multiline_response(self, mock):
        mock.return_value = _mock_response("Line 1\nLine 2\nLine 3")
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("multiline?")
        assert isinstance(result, str)
        assert "Line 2" in result


# ---------------------------------------------------------------------------
# response_model + append=True — history management
# ---------------------------------------------------------------------------


class TestResponseModelWithAppend:
    @patch("nerif.model.model.get_model_response")
    def test_append_true_keeps_history(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "A", "value": 1}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("first", append=True, response_model=SimpleItem)
        assert isinstance(result, SimpleItem)
        # History should include system + user + assistant
        assert len(model.messages) == 3
        assert model.messages[1]["role"] == "user"
        assert model.messages[2]["role"] == "assistant"

    @patch("nerif.model.model.get_model_response")
    def test_append_false_resets(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "A", "value": 1}))
        model = SimpleChatModel(model="gpt-4o")
        result = model.chat("first", append=False, response_model=SimpleItem)
        assert isinstance(result, SimpleItem)
        # History should be reset to just system message
        assert len(model.messages) == 1
        assert model.messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Async: response_model via achat()
# ---------------------------------------------------------------------------


class TestResponseModelAsync:
    def test_achat_returns_pydantic(self):
        async def run():
            model = SimpleChatModel(model="gpt-4o")
            mock_resp = _mock_response(json.dumps({"name": "AsyncWidget", "value": 99}))
            with patch("nerif.model.model.get_model_response_async", return_value=mock_resp):
                result = await model.achat("async item?", response_model=SimpleItem)
                assert isinstance(result, SimpleItem)
                assert result.name == "AsyncWidget"
                assert result.value == 99

        asyncio.get_event_loop().run_until_complete(run())

    def test_achat_without_model_returns_string(self):
        async def run():
            model = SimpleChatModel(model="gpt-4o")
            mock_resp = _mock_response("Just async text")
            with patch("nerif.model.model.get_model_response_async", return_value=mock_resp):
                result = await model.achat("hello")
                assert isinstance(result, str)
                assert result == "Just async text"

        asyncio.get_event_loop().run_until_complete(run())

    def test_achat_nested_model(self):
        async def run():
            data = {"name": "Eve", "age": 28, "address": {"street": "Oak", "city": "PDX"}}
            model = SimpleChatModel(model="gpt-4o")
            mock_resp = _mock_response(json.dumps(data))
            with patch("nerif.model.model.get_model_response_async", return_value=mock_resp):
                result = await model.achat("person?", response_model=PersonWithAddress)
                assert isinstance(result, PersonWithAddress)
                assert result.address.city == "PDX"

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# response_model + fallback — ensure parsing works after fallback
# ---------------------------------------------------------------------------


class TestResponseModelWithFallback:
    @patch("nerif.model.model.get_model_response")
    def test_fallback_then_parse(self, mock):
        """Fallback model returns valid JSON → pydantic parsing should succeed."""
        import httpx

        call_count = [0]

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            m = kwargs.get("model", "gpt-4o")
            if m == "gpt-4o":
                resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
                raise httpx.HTTPStatusError("fail", request=resp.request, response=resp)
            return _mock_response(json.dumps({"name": "Fallback", "value": 7}))

        mock.side_effect = side_effect
        model = SimpleChatModel(model="gpt-4o", fallback=["gpt-4o-mini"])
        result = model.chat("item?", response_model=SimpleItem)
        assert isinstance(result, SimpleItem)
        assert result.name == "Fallback"


# ---------------------------------------------------------------------------
# FormatVerifierPydantic edge cases
# ---------------------------------------------------------------------------


class TestFormatVerifierEdgeCases:
    def test_verify_with_dict_input(self):
        """verify() should handle dict input (not just string)."""
        v = FormatVerifierPydantic(SimpleItem)
        assert v.verify({"name": "X", "value": 1}) is True

    def test_convert_with_dict_input(self):
        v = FormatVerifierPydantic(SimpleItem)
        result = v.convert({"name": "X", "value": 1})
        assert isinstance(result, SimpleItem)

    def test_match_returns_none_for_garbage(self):
        v = FormatVerifierPydantic(SimpleItem)
        assert v.match("no json here whatsoever") is None

    def test_callable_interface(self):
        """FormatVerifierPydantic should be callable (used as verifier(text))."""
        v = FormatVerifierPydantic(SimpleItem)
        result = v(json.dumps({"name": "Z", "value": 0}))
        assert isinstance(result, SimpleItem)

    def test_cls_attribute(self):
        v = FormatVerifierPydantic(SimpleItem)
        assert v.cls is SimpleItem

    def test_empty_json_object(self):
        """Empty JSON {} should fail for model with required fields."""
        v = FormatVerifierPydantic(SimpleItem)
        assert v.verify("{}") is False

    def test_json_array_fails(self):
        """JSON array is not a valid pydantic model input."""
        v = FormatVerifierPydantic(SimpleItem)
        assert v.verify('[{"name": "X", "value": 1}]') is False


# ---------------------------------------------------------------------------
# Schema generation — ensure response_format is correctly built
# ---------------------------------------------------------------------------


class TestSchemaGeneration:
    @patch("nerif.model.model.get_model_response")
    def test_schema_sent_to_api(self, mock):
        mock.return_value = _mock_response(json.dumps({"name": "X", "value": 1}))
        model = SimpleChatModel(model="gpt-4o")
        model.chat("item?", response_model=SimpleItem)

        call_kwargs = mock.call_args[1]
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "SimpleItem"
        schema = rf["json_schema"]["schema"]
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]

    @patch("nerif.model.model.get_model_response")
    def test_nested_schema_includes_refs(self, mock):
        data = {"name": "A", "age": 1, "address": {"street": "s", "city": "c"}}
        mock.return_value = _mock_response(json.dumps(data))
        model = SimpleChatModel(model="gpt-4o")
        model.chat("person?", response_model=PersonWithAddress)

        call_kwargs = mock.call_args[1]
        rf = call_kwargs["response_format"]
        schema = rf["json_schema"]["schema"]
        # Nested model should appear in $defs or properties
        assert "NestedAddress" in str(schema)

    @patch("nerif.model.model.get_model_response")
    def test_no_schema_when_no_response_model(self, mock):
        mock.return_value = _mock_response("plain text")
        model = SimpleChatModel(model="gpt-4o")
        model.chat("hello")

        call_kwargs = mock.call_args[1]
        assert "response_format" not in call_kwargs
