import unittest
from unittest.mock import MagicMock, patch

from nerif.model import OllamaEmbeddingModel
from nerif.utils import NerifTokenCounter


class TestOllamaEmbeddingModel(unittest.TestCase):
    """Test cases for OllamaEmbeddingModel"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_text = "This is a test sentence for embedding."
        self.test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock embedding vector

    @patch('nerif.model.model.get_litellm_embedding')
    def test_ollama_embedding_default_model(self, mock_get_embedding):
        """Test OllamaEmbeddingModel with default model"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.data = [{"embedding": self.test_embedding}]
        mock_get_embedding.return_value = mock_response

        # Create model and test
        model = OllamaEmbeddingModel()
        result = model.embed(self.test_text)

        # Assertions
        self.assertEqual(result, self.test_embedding)
        mock_get_embedding.assert_called_once_with(
            messages=self.test_text,
            model="ollama/mxbai-embed-large",
            base_url="http://localhost:11434/v1/",
            counter=None,
        )

    @patch('nerif.model.model.get_litellm_embedding')
    def test_ollama_embedding_custom_model(self, mock_get_embedding):
        """Test OllamaEmbeddingModel with custom model"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.data = [{"embedding": self.test_embedding}]
        mock_get_embedding.return_value = mock_response

        # Create model with custom settings
        model = OllamaEmbeddingModel(
            model="ollama/nomic-embed-text",
            url="http://custom-ollama:11434/v1/"
        )
        result = model.embed(self.test_text)

        # Assertions
        self.assertEqual(result, self.test_embedding)
        mock_get_embedding.assert_called_once_with(
            messages=self.test_text,
            model="ollama/nomic-embed-text",
            base_url="http://custom-ollama:11434/v1/",
            counter=None,
        )

    @patch('nerif.model.model.get_litellm_embedding')
    def test_ollama_embedding_with_counter(self, mock_get_embedding):
        """Test OllamaEmbeddingModel with token counter"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.data = [{"embedding": self.test_embedding}]
        mock_get_embedding.return_value = mock_response

        # Create model with counter
        counter = NerifTokenCounter()
        model = OllamaEmbeddingModel(counter=counter)
        result = model.embed(self.test_text)

        # Assertions
        self.assertEqual(result, self.test_embedding)
        mock_get_embedding.assert_called_once_with(
            messages=self.test_text,
            model="ollama/mxbai-embed-large",
            base_url="http://localhost:11434/v1/",
            counter=counter,
        )

    @patch('nerif.model.model.get_litellm_embedding')
    def test_ollama_embedding_all_minilm(self, mock_get_embedding):
        """Test OllamaEmbeddingModel with all-minilm model"""
        # Setup mock
        mock_response = MagicMock()
        mock_response.data = [{"embedding": self.test_embedding}]
        mock_get_embedding.return_value = mock_response

        # Create model with all-minilm
        model = OllamaEmbeddingModel(model="ollama/all-minilm")
        result = model.embed(self.test_text)

        # Assertions
        self.assertEqual(result, self.test_embedding)
        mock_get_embedding.assert_called_once_with(
            messages=self.test_text,
            model="ollama/all-minilm",
            base_url="http://localhost:11434/v1/",
            counter=None,
        )


if __name__ == "__main__":
    unittest.main()