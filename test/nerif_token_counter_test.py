import pprint
import unittest

from litellm import completion, embedding

from nerif.core import nerif
from nerif.model import NerifTokenCounter, SimpleChatModel, SimpleEmbeddingModel

pretty_printer = pprint.PrettyPrinter()


class TestTokenCounter(unittest.TestCase):
    def test_counting_completion_raw(self):
        counter = NerifTokenCounter()
        response = completion(
            model="openrouter/openai/gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "Hello"}],
        )
        counter.count_from_response(response)
        response = completion(
            model="openrouter/openai/gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "Hello From World"}],
        )
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 1)
        self.assertGreater(
            counter.model_token["openrouter/openai/gpt-4o-2024-08-06"].request,
            0,
        )
        self.assertGreater(
            counter.model_token["openrouter/openai/gpt-4o-2024-08-06"].response,
            0,
        )

    def test_counting_embedding_raw(self):
        counter = NerifTokenCounter()
        response = embedding(model="text-embedding-3-small", input="Hello")
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 1)
        self.assertEqual(counter.model_token["text-embedding-3-small"].request, 0)
        self.assertGreater(counter.model_token["text-embedding-3-small"].response, 0)

    def test_counting_mixed_raw(self):
        counter = NerifTokenCounter()
        response = completion(
            model="openrouter/openai/gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "Hello"}],
        )
        counter.count_from_response(response)
        response = embedding(model="text-embedding-3-small", input="Hello")
        counter.count_from_response(response)
        response = completion(
            model="openrouter/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello From World"}],
        )
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 3)
        print(counter.model_token)

    def test_couting_agent(self):
        counter = NerifTokenCounter()
        chat_agent = SimpleChatModel(counter=counter)
        embedding_agent = SimpleEmbeddingModel(counter=counter)

        _ = chat_agent.chat("Which mobile suit is the most powerful one in Z Gundam.")
        self.assertEqual(len(counter.model_token.model_cost), 1)

        _ = embedding_agent.embed("Hello world")
        self.assertEqual(len(counter.model_token.model_cost), 2)

        _ = chat_agent.chat("Which mobile suit is the less powerful in Gundam NT")
        self.assertEqual(len(counter.model_token.model_cost), 2)
        print(counter.model_token)

    def test_couting_nerif(self):
        counter = NerifTokenCounter()
        if nerif("the sky is blue", counter=counter):
            print("True")

        self.assertEqual(len(counter.model_token.model_cost), 2)
        print(counter.model_token)

    def test_set_parser(self):
        counter = NerifTokenCounter()
        counter.set_parser_based_on_model("openrouter/openai/gpt-4o-2024-08-06")
        self.assertEqual(counter.response_parser.__class__.__name__, "OpenAIResponseParser")
        counter.set_parser_based_on_model("ollama/llama3.1")
        self.assertEqual(counter.response_parser.__class__.__name__, "OllamaResponseParser")
        counter.set_parser_based_on_model("openrouter/meta/llama3.1")
        self.assertEqual(counter.response_parser.__class__.__name__, "OpenAIResponseParser")


if __name__ == "__main__":
    unittest.main()
