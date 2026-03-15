"""Tests for PromptTemplate."""

import pytest

from nerif.utils.prompt import PromptTemplate


class TestBasicSubstitution:
    def test_single_variable(self):
        tpl = PromptTemplate("Hello {name}")
        assert tpl.format(name="World") == "Hello World"

    def test_multiple_variables(self):
        tpl = PromptTemplate("Summarize {text} in {language}")
        result = tpl.format(text="hello", language="Chinese")
        assert result == "Summarize hello in Chinese"

    def test_missing_variable_raises(self):
        tpl = PromptTemplate("Hello {name}")
        with pytest.raises(KeyError, match="Missing template variable: name"):
            tpl.format()

    def test_variable_converted_to_string(self):
        tpl = PromptTemplate("Count: {n}")
        assert tpl.format(n=42) == "Count: 42"


class TestDefaults:
    def test_default_used_when_not_overridden(self):
        tpl = PromptTemplate("Answer in {language}", defaults={"language": "English"})
        assert tpl.format() == "Answer in English"

    def test_default_overridden(self):
        tpl = PromptTemplate("Answer in {language}", defaults={"language": "English"})
        assert tpl.format(language="Chinese") == "Answer in Chinese"


class TestConditional:
    def test_conditional_included_when_variable_present(self):
        tpl = PromptTemplate("Do X.{? Use format: {format}}")
        assert tpl.format(format="JSON") == "Do X. Use format: JSON"

    def test_conditional_removed_when_variable_absent(self):
        tpl = PromptTemplate("Do X.{? Use format: {format}}")
        assert tpl.format() == "Do X."

    def test_conditional_removed_when_variable_is_none(self):
        tpl = PromptTemplate("Do X.{? Use format: {format}}")
        assert tpl.format(format=None) == "Do X."


class TestPartial:
    def test_partial_fills_defaults(self):
        tpl = PromptTemplate("Summarize {text} in {language}")
        partial = tpl.partial(language="Chinese")
        assert partial.format(text="hello") == "Summarize hello in Chinese"

    def test_partial_returns_new_template(self):
        tpl = PromptTemplate("Hello {name}")
        partial = tpl.partial(name="World")
        assert partial is not tpl


class TestVariablesProperty:
    def test_lists_all_variables(self):
        tpl = PromptTemplate("Hello {name}, you are {age}")
        assert tpl.variables == {"name", "age"}

    def test_includes_conditional_variables(self):
        tpl = PromptTemplate("Do X.{? format: {fmt}}")
        assert "fmt" in tpl.variables


class TestConcatenation:
    def test_add_combines_templates(self):
        a = PromptTemplate("Hello {name}. ")
        b = PromptTemplate("You are {age}.")
        combined = a + b
        assert combined.format(name="Bob", age=30) == "Hello Bob. You are 30."

    def test_add_merges_defaults(self):
        a = PromptTemplate("{a}", defaults={"a": "1"})
        b = PromptTemplate("{b}", defaults={"b": "2"})
        combined = a + b
        assert combined.format() == "12"


class TestRepr:
    def test_repr(self):
        tpl = PromptTemplate("Hello {name}")
        assert repr(tpl) == "PromptTemplate('Hello {name}')"
