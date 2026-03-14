"""Tests for Pydantic structured output integration (FormatVerifierPydantic, NerifFormat.pydantic_parse,
and SimpleChatModel.chat(response_model=...)).
"""

import json
import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from nerif.utils.format import FormatVerifierPydantic, NerifFormat

# ---------------------------------------------------------------------------
# Test Pydantic models
# ---------------------------------------------------------------------------


class City(BaseModel):
    name: str
    country: str
    population: int


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    age: int
    address: Optional[Address] = None
    tags: List[str] = []


# ---------------------------------------------------------------------------
# FormatVerifierPydantic – verify()
# ---------------------------------------------------------------------------


class TestFormatVerifierPydanticVerify(unittest.TestCase):
    def setUp(self):
        self.verifier = FormatVerifierPydantic(City)

    def test_verify_valid_json(self):
        valid = json.dumps({"name": "Paris", "country": "France", "population": 2161000})
        assert self.verifier.verify(valid) is True

    def test_verify_invalid_json_syntax(self):
        assert self.verifier.verify("not json at all") is False

    def test_verify_missing_required_field(self):
        missing = json.dumps({"name": "Paris", "country": "France"})
        assert self.verifier.verify(missing) is False

    def test_verify_wrong_type(self):
        wrong_type = json.dumps({"name": "Paris", "country": "France", "population": "many"})
        assert self.verifier.verify(wrong_type) is False


# ---------------------------------------------------------------------------
# FormatVerifierPydantic – match()
# ---------------------------------------------------------------------------


class TestFormatVerifierPydanticMatch(unittest.TestCase):
    def setUp(self):
        self.verifier = FormatVerifierPydantic(City)
        self.city_dict = {"name": "Tokyo", "country": "Japan", "population": 13960000}

    def test_match_plain_json(self):
        result = self.verifier.match(json.dumps(self.city_dict))
        assert isinstance(result, City)
        assert result.name == "Tokyo"

    def test_match_markdown_code_block(self):
        wrapped = "```json\n" + json.dumps(self.city_dict) + "\n```"
        result = self.verifier.match(wrapped)
        assert isinstance(result, City)
        assert result.country == "Japan"

    def test_match_surrounding_text(self):
        surrounded = "Here is the city info: " + json.dumps(self.city_dict) + " end."
        result = self.verifier.match(surrounded)
        assert isinstance(result, City)
        assert result.population == 13960000

    def test_match_invalid_returns_none(self):
        result = self.verifier.match("no json here at all, just words")
        assert result is None


# ---------------------------------------------------------------------------
# FormatVerifierPydantic – returns Pydantic instance
# ---------------------------------------------------------------------------


class TestFormatVerifierPydanticInstance(unittest.TestCase):
    def test_returns_pydantic_instance(self):
        verifier = FormatVerifierPydantic(City)
        data = json.dumps({"name": "Berlin", "country": "Germany", "population": 3769000})
        result = verifier(data)
        assert isinstance(result, City)
        assert result.name == "Berlin"


# ---------------------------------------------------------------------------
# Nested models, list fields, optional fields
# ---------------------------------------------------------------------------


class TestFormatVerifierPydanticComplexModels(unittest.TestCase):
    def test_nested_model(self):
        verifier = FormatVerifierPydantic(Person)
        data = json.dumps(
            {
                "name": "Alice",
                "age": 30,
                "address": {"street": "123 Main St", "city": "Springfield"},
                "tags": ["dev", "python"],
            }
        )
        result = verifier(data)
        assert isinstance(result, Person)
        assert isinstance(result.address, Address)
        assert result.address.city == "Springfield"
        assert result.tags == ["dev", "python"]

    def test_optional_field_absent(self):
        verifier = FormatVerifierPydantic(Person)
        data = json.dumps({"name": "Bob", "age": 25})
        result = verifier(data)
        assert isinstance(result, Person)
        assert result.address is None
        assert result.tags == []

    def test_list_field(self):
        verifier = FormatVerifierPydantic(Person)
        data = json.dumps({"name": "Carol", "age": 40, "tags": ["a", "b", "c"]})
        result = verifier(data)
        assert result.tags == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# FormatVerifierPydantic – convert()
# ---------------------------------------------------------------------------


class TestFormatVerifierPydanticConvert(unittest.TestCase):
    def test_convert_valid(self):
        verifier = FormatVerifierPydantic(City)
        data = json.dumps({"name": "Rome", "country": "Italy", "population": 2873000})
        result = verifier.convert(data)
        assert isinstance(result, City)
        assert result.name == "Rome"

    def test_convert_invalid_raises(self):
        verifier = FormatVerifierPydantic(City)
        with pytest.raises(Exception):
            verifier.convert(json.dumps({"name": "Rome"}))


# ---------------------------------------------------------------------------
# NerifFormat.pydantic_parse
# ---------------------------------------------------------------------------


class TestNerifFormatPydanticParse(unittest.TestCase):
    def test_pydantic_parse_plain_json(self):
        data = json.dumps({"name": "Madrid", "country": "Spain", "population": 3300000})
        result = NerifFormat.pydantic_parse(data, City)
        assert isinstance(result, City)
        assert result.name == "Madrid"

    def test_pydantic_parse_markdown_wrapped(self):
        city_dict = {"name": "Lisbon", "country": "Portugal", "population": 504718}
        wrapped = "```json\n" + json.dumps(city_dict) + "\n```"
        result = NerifFormat.pydantic_parse(wrapped, City)
        assert isinstance(result, City)
        assert result.country == "Portugal"

    def test_pydantic_parse_invalid_raises(self):
        with pytest.raises(Exception):
            NerifFormat.pydantic_parse("completely invalid", City)


# ---------------------------------------------------------------------------
# SimpleChatModel.chat(response_model=...)  –  mocked httpx
# ---------------------------------------------------------------------------


def _make_mock_response(content: str):
    """Build a minimal mock that mimics ChatCompletionResponse structure."""
    tool_call_mock = MagicMock()
    tool_call_mock.tool_calls = None

    message_mock = MagicMock()
    message_mock.content = content
    message_mock.tool_calls = None

    choice_mock = MagicMock()
    choice_mock.message = message_mock

    response_mock = MagicMock()
    response_mock.choices = [choice_mock]
    return response_mock


class TestSimpleChatModelResponseModel(unittest.TestCase):
    def _get_mock_model(self):
        from nerif.model.model import SimpleChatModel

        return SimpleChatModel(model="gpt-4o")

    @patch("nerif.model.model.get_model_response")
    def test_response_model_returns_pydantic_instance(self, mock_get_model_response):
        city_json = json.dumps({"name": "Vienna", "country": "Austria", "population": 1911000})
        mock_get_model_response.return_value = _make_mock_response(city_json)

        model = self._get_mock_model()
        result = model.chat("Tell me about Vienna", response_model=City)

        assert isinstance(result, City)
        assert result.name == "Vienna"
        assert result.country == "Austria"

    @patch("nerif.model.model.get_model_response")
    def test_response_model_sets_response_format_in_kwargs(self, mock_get_model_response):
        city_json = json.dumps({"name": "Prague", "country": "Czech Republic", "population": 1309000})
        mock_get_model_response.return_value = _make_mock_response(city_json)

        model = self._get_mock_model()
        model.chat("Tell me about Prague", response_model=City)

        call_kwargs = mock_get_model_response.call_args
        # response_format should be injected into kwargs
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        assert "response_format" in kwargs
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "City"

    @patch("nerif.model.model.get_model_response")
    def test_response_model_none_returns_string(self, mock_get_model_response):
        mock_get_model_response.return_value = _make_mock_response("Hello world")

        model = self._get_mock_model()
        result = model.chat("Say hello", response_model=None)

        assert isinstance(result, str)
        assert result == "Hello world"

    @patch("nerif.model.model.get_model_response")
    def test_response_model_nested(self, mock_get_model_response):
        person_json = json.dumps(
            {
                "name": "Eve",
                "age": 28,
                "address": {"street": "456 Oak Ave", "city": "Portland"},
                "tags": ["engineer"],
            }
        )
        mock_get_model_response.return_value = _make_mock_response(person_json)

        model = self._get_mock_model()
        result = model.chat("Tell me about Eve", response_model=Person)

        assert isinstance(result, Person)
        assert isinstance(result.address, Address)
        assert result.address.city == "Portland"


if __name__ == "__main__":
    unittest.main()
