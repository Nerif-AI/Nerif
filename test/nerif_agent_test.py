import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nerif.agent import LogitsAgent, SimpleChatAgent


class TestNerifAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")

    def test_simple_chat_agent(self):
        simple_agent = SimpleChatAgent()
        result = simple_agent.chat("Hello, how are you?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_agent_reset(self):
        simple_agent = SimpleChatAgent()
        simple_agent.chat("Remember this: blue sky")
        simple_agent.reset()
        result = simple_agent.chat("What did I ask you to remember?")
        self.assertNotIn("blue sky", result.lower())

    def test_agent_temperature(self):
        simple_agent1 = SimpleChatAgent(temperature=0.7)
        simple_agent2 = SimpleChatAgent(temperature=0.7)
        result1 = simple_agent1.chat("Generate a random number")
        result2 = simple_agent2.chat("Generate a random number")
        self.assertNotEqual(result1, result2)

    def test_logits_agent(self):
        logits_agent = LogitsAgent()
        result = logits_agent.chat("Hello, how are you?", max_tokens=1)
        print(result)
        self.assertIsNotNone(result)

    def test_ollama_agent(self):
        ollama_agent = SimpleChatAgent(model="ollama/llama3.1")
        result = ollama_agent.chat("Hello, how are you?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
