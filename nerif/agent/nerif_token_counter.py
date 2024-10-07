import uuid
from dataclasses import dataclass
from typing import Dict, List

import tiktoken

from .utils import OPENAI_EMBEDDING_MODEL, OPENAI_MODEL


class ModelCost:
    def __init__(self, model_name, request=0, response=0):
        self.model_name = model_name
        self.request = request
        self.response = response

    def add_cost(self, request, response=0):
        self.request += request
        self.response += response

    def __repr__(self) -> str:
        return f"{self.model_name}: {self.request} tokens requested, {self.response} tokens returned"


class NerifTokenConsume:

    def __init__(self):
        self.model_cost = {}

    def __getitem__(self, key):
        return self.model_cost[key]

    def append(self, consume: ModelCost):
        if consume is not None:
            model_name = consume.model_name
            if self.model_cost.get(model_name) is None:
                self.model_cost[model_name] = ModelCost(model_name)
            self.model_cost[model_name].add_cost(consume.request, consume.response)

        return self

    def __repr__(self) -> str:
        return str(self.model_cost)


class ResponseParserBase:
    def __call__(self, response) -> NerifTokenConsume:
        raise NotImplementedError("ResponseParserBase __call__ is not implemented")


class OpenAIResponseParser(ResponseParserBase):
    def __call__(self, response) -> NerifTokenConsume:
        model_name = response.model
        response_type = response.__class__.__name__
        if response_type == "EmbeddingResponse":
            requested_tokens = 0
            completation_tokens = len(response.data[0]["embedding"])
        else:
            usage = response.usage
            requested_tokens = usage.prompt_tokens
            completation_tokens = usage.completion_tokens

        consume = ModelCost(model_name, requested_tokens, completation_tokens)
        return consume


class NerifTokenCounter:
    """
    Class for counting tokens consumed by the model
    members:
    - model_token: Dict[(str, uuid.UUID), NerifTokenConsume] - dictionary for storing token consumption
    """

    def __init__(self, response_parser: ResponseParserBase = OpenAIResponseParser()):
        """
        Class for counting tokens consumed by the model
        """
        self.model_token = NerifTokenConsume()
        self.response_parser = response_parser

    def set_parser(self, parser: ResponseParserBase):
        self.response_parser = parser

    def count_from_response(self, response):
        """
        Counting tokens consumed by the model from response

        paramaters:
        - response: any - response from the model
        """
        consume = self.response_parser(response)
        self.model_token.append(consume)
