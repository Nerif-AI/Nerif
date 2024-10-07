import unittest

from litellm import completion, embedding

from nerif.agent.nerif_token_counter import NerifTokenCounter


class TestTokenCounter(unittest.TestCase):

    def test_counting_completion(self):
        counter = NerifTokenCounter()
        response = completion(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": "Hello"}])
        counter.count_from_response(response)
        response = completion(model="gpt-4o-2024-08-06", messages=[{"role": "user", "content": "Hello From World"}])
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 1)
        self.assertGreater(counter.model_token["gpt-4o-2024-08-06"].request, 0)
        self.assertGreater(counter.model_token["gpt-4o-2024-08-06"].response, 0)

    def test_counting_embedding(self):
        counter = NerifTokenCounter()
        response = embedding(model="text-embedding-3-small", input="Hello")
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 1)
        self.assertEqual(counter.model_token["text-embedding-3-small"].request, 0)
        self.assertGreater(counter.model_token["text-embedding-3-small"].response, 0)

    def test_counting_mixed(self):
        counter = NerifTokenCounter()
        response = completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])
        counter.count_from_response(response)
        response = embedding(model="text-embedding-3-small", input="Hello")
        counter.count_from_response(response)
        response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello From World"}])
        counter.count_from_response(response)

        self.assertEqual(len(counter.model_token.model_cost), 3)
        print(counter.model_token)


if __name__ == "__main__":
    unittest.main()
