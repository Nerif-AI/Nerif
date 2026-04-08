import os
import pprint
import unittest

from nerif.core import nerif
from nerif.model import SimpleChatModel, SimpleEmbeddingModel
from nerif.utils import NerifTokenCounter, get_embedding, get_model_response

pretty_printer = pprint.PrettyPrinter()
OPENROUTER_TEST_MODEL = "openrouter/xiaomi/mimo-v2-pro"
OPENAI_TEST_MODEL = "gpt-4o-mini"

_RUN_LIVE = bool(os.environ.get("NERIF_RUN_LIVE_TESTS"))


class TestTokenCounter(unittest.TestCase):
    @unittest.skipUnless(_RUN_LIVE, "NERIF_RUN_LIVE_TESTS not set — skipping live API tests")
    def test_counting_completion_raw(self):
        counter = NerifTokenCounter()
        response = get_model_response(
            model=OPENROUTER_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
        )
        completion_model_name = response.model
        counter.count_from_response(response)
        response = get_model_response(
            model=OPENROUTER_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello From World"}],
            max_tokens=16,
        )
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 1)
        self.assertGreater(counter.model_token[completion_model_name].request, 0)
        self.assertGreater(counter.model_token[completion_model_name].response, 0)

    @unittest.skipUnless(_RUN_LIVE, "NERIF_RUN_LIVE_TESTS not set — skipping live API tests")
    def test_counting_embedding_raw(self):
        counter = NerifTokenCounter()
        response = get_embedding(model="text-embedding-3-small", messages="Hello")
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 1)
        self.assertEqual(counter.model_token["text-embedding-3-small"].request, 0)
        self.assertGreater(counter.model_token["text-embedding-3-small"].response, 0)

    @unittest.skipUnless(_RUN_LIVE, "NERIF_RUN_LIVE_TESTS not set — skipping live API tests")
    def test_counting_mixed_raw(self):
        counter = NerifTokenCounter()
        response = get_model_response(
            model=OPENROUTER_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
        )
        counter.count_from_response(response)
        response = get_embedding(model="text-embedding-3-small", messages="Hello")
        counter.count_from_response(response)
        response = get_model_response(
            model=OPENAI_TEST_MODEL,
            messages=[{"role": "user", "content": "Hello From World"}],
            max_tokens=16,
        )
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 3)
        print(counter.model_token)

    @unittest.skipUnless(_RUN_LIVE, "NERIF_RUN_LIVE_TESTS not set — skipping live API tests")
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

    @unittest.skipUnless(_RUN_LIVE, "NERIF_RUN_LIVE_TESTS not set — skipping live API tests")
    def test_couting_nerif(self):
        counter = NerifTokenCounter()
        if nerif("the sky is blue", counter=counter):
            print("True")

        self.assertEqual(len(counter.model_token.model_cost), 2)
        print(counter.model_token)

    def test_set_parser(self):
        counter = NerifTokenCounter()
        counter.set_parser_based_on_model(OPENROUTER_TEST_MODEL)
        self.assertEqual(counter.response_parser.__class__.__name__, "OpenAIResponseParser")
        counter.set_parser_based_on_model("ollama/llama3.1")
        self.assertEqual(counter.response_parser.__class__.__name__, "OllamaResponseParser")
        counter.set_parser_based_on_model("openrouter/meta/llama3.1")
        self.assertEqual(counter.response_parser.__class__.__name__, "OpenAIResponseParser")


if __name__ == "__main__":
    unittest.main()
