import pytest

from nerif.utils.format import FormatVerifierStringList


@pytest.fixture
def verifier():
    return FormatVerifierStringList()


class TestVerify:
    def test_verify_valid_list(self, verifier):
        assert verifier.verify('["a", "b"]') is True

    def test_verify_not_a_list(self, verifier):
        assert verifier.verify("not a list") is False


class TestMatch:
    def test_match_python_list_string(self, verifier):
        result = verifier.match('["hello", "world"]')
        assert result is not None
        assert "hello" in result
        assert "world" in result

    def test_match_markdown_bullet_points(self, verifier):
        result = verifier.match("- item1\n- item2\n- item3")
        assert result is not None
        assert len(result) == 3
        assert result[0] == "item1"
        assert result[1] == "item2"
        assert result[2] == "item3"

    def test_match_numbered_list(self, verifier):
        result = verifier.match("1. first\n2. second")
        assert result is not None
        assert len(result) == 2

    def test_match_empty_no_match(self, verifier):
        result = verifier.match("")
        assert result is None


class TestConvert:
    def test_convert_valid_python_list(self, verifier):
        result = verifier.convert('["apple", "banana"]')
        assert isinstance(result, list)
        assert result == ["apple", "banana"]


class TestCall:
    def test_call_valid_list_string(self, verifier):
        result = verifier('["x", "y", "z"]')
        assert isinstance(result, list)
        assert result == ["x", "y", "z"]
