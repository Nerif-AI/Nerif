import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nerif.model import SimpleChatModel, SimpleEmbeddingModel


class TestNerifAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")

    def test_ollama_agent(self):
        ollama_agent = SimpleChatModel(model="ollama/llama3.1")
        result = ollama_agent.chat("Hello, how are you?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_ollama_embed_agent(self):
        ollama_agent = SimpleEmbeddingModel(model="ollama/mxbai-embed-large")
        result = ollama_agent.embed("Hello, how are you?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list[float])
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
