"""Tests for optional embedding model support (Feature 6)."""

import unittest
from unittest.mock import MagicMock, patch


class TestNerificationBaseLazyInit(unittest.TestCase):
    """Test NerificationBase lazy embedding initialization."""

    def test_embedding_not_created_on_init(self):
        """Embedding model should not be instantiated during __init__."""
        from nerif.core.core import NerificationBase

        with patch("nerif.core.core.SimpleEmbeddingModel") as mock_emb:
            base = NerificationBase(possible_values=["a", "b"], model="some-model")
            mock_emb.assert_not_called()
            self.assertIsNone(base._embedding)

    def test_embedding_created_on_first_access_when_model_set(self):
        """Accessing .embedding should create SimpleEmbeddingModel lazily."""
        from nerif.core.core import NerificationBase

        mock_instance = MagicMock()
        with patch("nerif.core.core.SimpleEmbeddingModel", return_value=mock_instance) as mock_cls:
            base = NerificationBase(possible_values=["a", "b"], model="some-model")
            mock_cls.assert_not_called()

            emb = base.embedding
            mock_cls.assert_called_once_with(model="some-model", counter=None)
            self.assertIs(emb, mock_instance)

    def test_embedding_cached_after_first_access(self):
        """Second access to .embedding should return the cached instance."""
        from nerif.core.core import NerificationBase

        mock_instance = MagicMock()
        with patch("nerif.core.core.SimpleEmbeddingModel", return_value=mock_instance) as mock_cls:
            base = NerificationBase(possible_values=["a", "b"], model="some-model")
            _ = base.embedding
            _ = base.embedding
            # Only one instantiation despite two accesses
            self.assertEqual(mock_cls.call_count, 1)

    def test_embedding_raises_when_no_model(self):
        """Accessing .embedding without a model should raise RuntimeError."""
        from nerif.core.core import NerificationBase

        base = NerificationBase(possible_values=["a", "b"], model=None)
        with self.assertRaises(RuntimeError) as ctx:
            _ = base.embedding
        self.assertIn("Embedding model not configured", str(ctx.exception))

    def test_has_embedding_true_with_model(self):
        """has_embedding should be True when a model name is set."""
        from nerif.core.core import NerificationBase

        base = NerificationBase(possible_values=["a", "b"], model="text-embedding-3-small")
        self.assertTrue(base.has_embedding)

    def test_has_embedding_false_with_none(self):
        """has_embedding should be False when model is None."""
        from nerif.core.core import NerificationBase

        base = NerificationBase(possible_values=["a", "b"], model=None)
        self.assertFalse(base.has_embedding)

    def test_has_embedding_false_with_empty_string(self):
        """has_embedding should be False when model is empty string."""
        from nerif.core.core import NerificationBase

        base = NerificationBase(possible_values=["a", "b"], model="")
        self.assertFalse(base.has_embedding)


class TestNerificationIntCounterBugFix(unittest.TestCase):
    """Verify NerificationInt passes counter to super().__init__."""

    def test_counter_passed_to_base(self):
        """NerificationInt should pass counter to NerificationBase."""
        from nerif.core.core import NerificationInt

        mock_counter = MagicMock()
        with patch("nerif.core.core.SimpleEmbeddingModel"):
            inst = NerificationInt(possible_values=[0, 1, 2], model="m", counter=mock_counter)
            self.assertIs(inst._counter, mock_counter)


class TestNerifJudgeLogitsSuccess(unittest.TestCase):
    """judge() should return early from logits mode without touching embedding."""

    def test_logits_success_no_embedding_needed(self):
        """When logits mode succeeds, embedding is never accessed."""
        from nerif.core.core import Nerif

        fake_logprobs_response = MagicMock()
        fake_logprobs_response.choices = [MagicMock()]
        fake_logprobs_response.choices[0].logprobs = {
            "content": [{"top_logprobs": [{"token": "True", "logprob": -0.1}]}]
        }

        with (
            patch("nerif.core.core.SimpleChatModel"),
            patch("nerif.core.core.LogitsChatModel") as mock_logits_cls,
            patch("nerif.core.core.support_logit_mode", return_value=True),
        ):
            mock_logits_instance = MagicMock()
            mock_logits_instance.chat.return_value = fake_logprobs_response
            mock_logits_cls.return_value = mock_logits_instance

            nerif_inst = Nerif(model="gpt-4o", embed_model=None)
            # Ensure embedding property is never triggered
            with patch.object(
                type(nerif_inst.verification),
                "embedding",
                new_callable=lambda: property(
                    lambda self: (_ for _ in ()).throw(AssertionError("embedding accessed unexpectedly"))
                ),
            ):
                result = nerif_inst.judge("the sky is blue")

        self.assertIsNotNone(result)


class TestNerifTextFallback(unittest.TestCase):
    """Nerif.text_fallback_mode() should return a bool."""

    def _make_nerif_no_embed(self, chat_response: str):
        from nerif.core.core import Nerif

        with patch("nerif.core.core.SimpleChatModel") as mock_simple_cls, patch("nerif.core.core.LogitsChatModel"):
            mock_agent = MagicMock()
            mock_agent.chat.return_value = chat_response
            mock_simple_cls.return_value = mock_agent
            inst = Nerif(model="gpt-4o", embed_model=None)
            inst.agent = mock_agent
        return inst

    def test_text_fallback_returns_true_for_true_response(self):
        inst = self._make_nerif_no_embed("True")
        result = inst.text_fallback_mode("the sky is blue")
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_text_fallback_returns_false_for_false_response(self):
        inst = self._make_nerif_no_embed("False")
        result = inst.text_fallback_mode("pigs can fly")
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_text_fallback_lowercase_true(self):
        inst = self._make_nerif_no_embed("true, definitely")
        result = inst.text_fallback_mode("water is wet")
        self.assertIsInstance(result, bool)

    def test_judge_uses_text_fallback_when_no_embedding(self):
        """judge() should call text_fallback_mode when has_embedding is False."""
        from nerif.core.core import Nerif

        with (
            patch("nerif.core.core.SimpleChatModel") as mock_simple_cls,
            patch("nerif.core.core.LogitsChatModel"),
            patch("nerif.core.core.support_logit_mode", return_value=False),
        ):
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "True"
            mock_simple_cls.return_value = mock_agent

            inst = Nerif(model="some-model", embed_model=None)
            inst.agent = mock_agent

            with patch.object(inst, "text_fallback_mode", return_value=True) as mock_fallback:
                result = inst.judge("some question")

        mock_fallback.assert_called_once_with("some question")
        self.assertTrue(result)


class TestNerifMatchStringTextFallback(unittest.TestCase):
    """NerifMatchString.text_fallback_mode() should return a valid index."""

    def _make_match_no_embed(self, choices, chat_response: str):
        from nerif.core.core import NerifMatchString

        with patch("nerif.core.core.SimpleChatModel") as mock_simple_cls, patch("nerif.core.core.LogitsChatModel"):
            mock_agent = MagicMock()
            mock_agent.chat.return_value = chat_response
            mock_simple_cls.return_value = mock_agent
            inst = NerifMatchString(choices=choices, model="gpt-4o", embed_model=None)
            inst.agent = mock_agent
        return inst

    def test_text_fallback_returns_int_on_simple_match(self):
        choices = ["apple", "banana", "cherry"]
        inst = self._make_match_no_embed(choices, "I think option 1 is best")
        result = inst.text_fallback_mode("which fruit is yellow?")
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)

    def test_text_fallback_default_zero_on_no_match(self):
        choices = ["option A", "option B", "option C"]
        inst = self._make_match_no_embed(choices, "I have no idea whatsoever xyz")
        result = inst.text_fallback_mode("random question")
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)

    def test_match_uses_text_fallback_when_no_embedding(self):
        """match() should call text_fallback_mode when has_embedding is False."""
        from nerif.core.core import NerifMatchString

        with (
            patch("nerif.core.core.SimpleChatModel") as mock_simple_cls,
            patch("nerif.core.core.LogitsChatModel"),
            patch("nerif.core.core.support_logit_mode", return_value=False),
        ):
            mock_agent = MagicMock()
            mock_agent.chat.return_value = "1"
            mock_simple_cls.return_value = mock_agent

            inst = NerifMatchString(choices=["a", "b", "c"], model="some-model", embed_model=None)
            inst.agent = mock_agent

            with patch.object(inst, "text_fallback_mode", return_value=1) as mock_fallback:
                result = inst.match("some question")

        mock_fallback.assert_called_once_with("some question")
        self.assertEqual(result, 1)


class TestExistingBehaviorUnchangedWithEmbedding(unittest.TestCase):
    """Verify existing code paths still work when embedding IS available."""

    def test_nerif_judge_uses_embedding_mode_when_available(self):
        """When embedding is available and logits fails, embedding_mode is called."""
        from nerif.core.core import Nerif

        with (
            patch("nerif.core.core.SimpleChatModel"),
            patch("nerif.core.core.LogitsChatModel"),
            patch("nerif.core.core.support_logit_mode", return_value=False),
        ):
            inst = Nerif(model="some-model", embed_model="text-embedding-3-small")

            with patch.object(inst, "embedding_mode", return_value=True) as mock_emb_mode:
                result = inst.judge("the sky is blue")

        mock_emb_mode.assert_called_once_with("the sky is blue")
        self.assertTrue(result)

    def test_nerif_match_uses_embedding_mode_when_available(self):
        """When embedding is available and logits fails, embedding_mode is called."""
        from nerif.core.core import NerifMatchString

        with (
            patch("nerif.core.core.SimpleChatModel"),
            patch("nerif.core.core.LogitsChatModel"),
            patch("nerif.core.core.support_logit_mode", return_value=False),
        ):
            inst = NerifMatchString(choices=["a", "b"], model="some-model", embed_model="text-embedding-3-small")

            with patch.object(inst, "embedding_mode", return_value=1) as mock_emb_mode:
                result = inst.match("some question")

        mock_emb_mode.assert_called_once_with("some question")
        self.assertEqual(result, 1)


class TestEmptyStringEnvVarDisablesEmbedding(unittest.TestCase):
    """Empty string NERIF_DEFAULT_EMBEDDING_MODEL should disable embedding."""

    def test_empty_env_var_sets_none(self):
        """When env var is empty string, NERIF_DEFAULT_EMBEDDING_MODEL becomes None."""
        import importlib
        import os

        with patch.dict(os.environ, {"NERIF_DEFAULT_EMBEDDING_MODEL": ""}):
            import nerif.utils.utils as utils_module

            importlib.reload(utils_module)
            self.assertIsNone(utils_module.NERIF_DEFAULT_EMBEDDING_MODEL)

    def test_set_env_var_preserved_as_model_name(self):
        """When env var has a real model name, it's preserved."""
        import importlib
        import os

        with patch.dict(os.environ, {"NERIF_DEFAULT_EMBEDDING_MODEL": "text-embedding-3-large"}):
            import nerif.utils.utils as utils_module

            importlib.reload(utils_module)
            self.assertEqual(utils_module.NERIF_DEFAULT_EMBEDDING_MODEL, "text-embedding-3-large")


if __name__ == "__main__":
    unittest.main()
