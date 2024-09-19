import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel

from nerif.nerif_agent.nerif_agent import (
    LogitsAgent,
    SimpleChatAgent,
    SimpleEmbeddingAgent,
    StructuredAgent
)

print("Testing model {}".format(os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")))


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
    
    def test_structured_agent(self):
        import litellm
        litellm.enable_json_schema_validation = True
        litellm.set_verbose = True # see the raw request made by litellm
        class ResponseFormat(BaseModel):
            year: int
            model: str
            cpu_name: str
        structured_agent = StructuredAgent(model="gpt-4o-2024-08-06")
        result = structured_agent.chat("Which iPhone is published at 2012? Reply in json.", response_format=ResponseFormat)
        # result = structured_agent.chat("Which iPhone is published at 2012? Fill following json {\"year\": <>, \"model name\": <>}", response_format={"type": "json_object"})
        print(result)
        self.assertIsNotNone(result)
        self.assertIn("iphone 5", result["model"].lower())
        self.assertIn("A6", result["cpu_name"])
        self.assertEqual(result["year"], 2012)
    
    def test_ollama_agent(self):
        ollama_agent = SimpleChatAgent(model="ollama/llama3.1")
        result = ollama_agent.chat("Hello, how are you?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
