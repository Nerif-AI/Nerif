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
            published_year: int
            model: str
            cpu_name: str
            exist: bool
        structured_agent = StructuredAgent(model="gpt-4o-2024-08-06")
        result = structured_agent.chat("Is iPhone XX exist? If exist, tell me the cpu name and its publish year, or just reply me not exit.", response_format=ResponseFormat)
        print(result)
        choice = result.choices[0]
        self.assertIsNotNone(choice)
        content = ResponseFormat.model_validate_json(choice.message.content)
        self.assertIn("iphone xx", content.model.lower())
        self.assertEqual("", content.cpu_name)
        self.assertEqual(content.published_year, 0)
        class ResponseFormat(BaseModel):
            number: int
            upper_bound: int
            lower_bound: int
        structured_agent = StructuredAgent(model="gpt-4o-2024-08-06")
        result = structured_agent.chat("Generate a random number between 1 to 100. Reply in json.", response_format=ResponseFormat)
        choice = result.choices[0]
        self.assertIsNotNone(choice)
        content = ResponseFormat.model_validate_json(choice.message.content)
        self.assertIsInstance(content.number, int)
        self.assertGreaterEqual(content.number, content.lower_bound)
        self.assertLessEqual(content.number, content.upper_bound)
        
        
    
    def test_ollama_agent(self):
        ollama_agent = SimpleChatAgent(model="ollama/llama3.1")
        result = ollama_agent.chat("Hello, how are you?")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
