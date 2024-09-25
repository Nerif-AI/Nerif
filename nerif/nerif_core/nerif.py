import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from openai import OpenAI

from nerif.nerif_agent.nerif_agent import (
    LogitsAgent,
    SimpleChatAgent,
    SimpleEmbeddingAgent,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
NERIF_DEFAULT_LLM_MODEL = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
NERIF_DEFAULT_EMBEDDING_MODEL = os.environ.get("NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")


def similarity_dist(vec1, vec2, func="cosine"):
    if func == "cosine":
        return 1 - (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return np.linalg.norm(vec1 - vec2)


class NerificationBase:
    """
    Base class for Nerification.
    This class is used to verify the result of the Nerif.
    """

    def __init__(self, possible_values: Optional[List[Any]] = None, model: str = NERIF_DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the NerificationBase.
        possible_values: list[Any] = None
        model: str = NERIF_DEFAULT_EMBEDDING_MODEL

        possible_values: list of possible values to verify. Only store lower case string
        """
        self.original_options = possible_values
        # Convert the possible value to lower case
        self.possible = []

        for index in range(len(possible_values)):
            self.possible.append(self.convert(possible_values[index]))

        self.embedding = SimpleEmbeddingAgent(model=model)
        self.possible_embed = []

        # Embed the possible value and the possible instruction
        for index in range(len(self.possible)):
            self.possible_embed.append(self.embedding.encode(self.possible[index]))

    def convert(self, val: Any):
        """
        Convert the value to lower case.
        """
        if isinstance(val, str):
            return val.lower()
        else:
            return str(val).lower()

    def verify(self, val: Any):
        """
        Verify if the text is in the possible_values.
        """
        if self.convert(val) in self.possible:
            return True
        return False

    def simple_fit(self, val: Any):
        """
        Use the embedding model to find the best fit in the possible_values.
        """
        text = self.convert(val)
        for index in range(len(self.possible)):
            if self.possible[index] in text:
                return self.original_options[index]
        return None

    def force_fit(self, val: Any, similarity="cosine"):
        """
        Use the embedding model to find the best fit in the possible_values.
        """
        text_embed = self.embedding.encode(self.convert(val))
        min_dist = similarity_dist(text_embed, self.possible_embed[0], similarity)
        min_id = 0
        for index in range(1, len(self.possible_embed)):
            dist = similarity_dist(text_embed, self.possible_embed[index], similarity)
            if dist < min_dist:
                min_dist = dist
                min_id = index
        return self.original_options[min_id]


# * How to build child class for NerificationBase:
# * __init__: possible_values, model
# * - convert possible_values to lower case string (implemented in convert method), store to `possible`
# * - store the original possible_values to `original_options`
# * - create a SimpleEmbeddingAgent with the model to `embedding`
# * - store embedded possible_values to `possible_embed`
# * convert: convert the value to lower case
# * verify: check if the value is in the possible_values
# * simple_fit: find the best fit in the possible_values
# * force_fit: force find the best fit in possible_values using embedding


class Nerification(NerificationBase):
    """
    Bool Value Verification
    """

    def __init__(
        self,
        model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    ):
        super().__init__([True, False], model)

    def convert(self, val: Any):
        return str(val).lower()

    def verify(self, val: Any):
        return super().verify(val)

    def simple_fit(self, val: Any):
        return super().simple_fit(val)

    def force_fit(self, val: Any, similarity="cosine"):
        return super().force_fit(val, similarity)


class NerificationString(NerificationBase):
    """
    String Value Verification
    This class is used to verify the result of the Nerif.
    If the result is in the possible_values, return the item.
    If the result is not in the possible_values, return None.
    """

    def __init__(self, possible_values: Optional[List[str]] = None, model: str = NERIF_DEFAULT_EMBEDDING_MODEL):
        super().__init__(possible_values, model)

    def convert(self, val: Any):
        return val.lower()

    def verify(self, val: Any):
        return super().verify(val)

    def simple_fit(self, val: Any):
        return super().simple_fit(val)

    def force_fit(self, val: Any, similarity="cosine"):
        return super().force_fit(val, similarity)


class NerificationInt(NerificationBase):
    """
    Int Value Verification
    """

    def __init__(self, possible_values: Optional[List[int]] = None, model: str = NERIF_DEFAULT_EMBEDDING_MODEL):
        super().__init__(possible_values, model)

    def convert(self, val: Any):
        return str(val)

    def verify(self, val: Any):
        return super().verify(val)

    def simple_fit(self, val: Any):
        return super().simple_fit(val)

    def force_fit(self, val: Any, similarity="cosine"):
        return super().force_fit(val, similarity)


class Nerif:
    """
    This class is used to judge the truthfulness of a statement.
    It uses two modes: logits mode and embedding mode.
    Logits mode is faster, but less accurate. Fetch top logprobs from the logits.
    Embedding mode is slower, but more accurate. Embed the text and compare with the possible values.

    Attributes:
        model: str = NERIF_DEFAULT_LLM_MODEL
        temperature: float = 0
        debug: bool = False

    Methods:
        logits_mode(text: str) -> bool:
            Judge the truthfulness of the statement using logits mode.
        embedding_mode(text: str) -> bool:
            Judge the truthfulness of the statement using embedding mode.
        judge(text: str, max_retry: int = 3) -> bool:
            Judge the truthfulness of the statement.
        instance(text: str, max_retry: int = 3, model: str = NERIF_DEFAULT_LLM_MODEL, debug: bool = False) -> bool:
            Judge the truthfulness of the statement.
    """

    def __init__(self, model=NERIF_DEFAULT_LLM_MODEL, temperature=0, debug=False):
        self.model = model
        self.prompt = (
            "Given the following text, determine if the statement is true or false.\n"
            "<question>\n"
            "Only answer with 'True' or 'False'."
        )
        self.temperature = temperature
        self.agent = SimpleChatAgent(model=model, temperature=temperature)
        self.logits_agent = LogitsAgent(model=model, temperature=temperature)
        self.verification = Nerification()
        self.debug = debug

        if self.debug:
            print("Nerif initialized with model:", model, "temperature:", temperature, "debug:", debug)

    def logits_mode(self, text: str):
        if self.debug:
            print("Logits mode, text:", text)
        self.logits_agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.logits_agent.chat(user_prompt, max_tokens=1)
        if self.debug:
            print("Logits mode, response:", response)
        if not(hasattr(response, 'choices') and len(response.choices) > 0):
            raise AttributeError("'Response' object has no attribute 'choices'")
        # if choices doesn't have no logprobs, raise an exception
        if not hasattr(response.choices[0], 'logprobs') or response.choices[0].logprobs is None:
            raise AttributeError("'Choices' object has no attribute 'logprobs'")
        logprobs = response.choices[0].logprobs["content"][0]
        sorted_logprobs = sorted(logprobs["top_logprobs"], key=lambda x: x["logprob"], reverse=True)
        # Try to find the most likely logprob
        for index in range(len(sorted_logprobs)):
            if self.debug:
                print("Logits mode, sorted_logprobs[index]:", sorted_logprobs[index]["token"])
            simple_fit = self.verification.simple_fit(sorted_logprobs[index]["token"])
            if simple_fit is not None:
                return simple_fit
        return None

    def embedding_mode(self, text: str):
        if self.debug:
            print("Embedding mode, text:", text)
        self.agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.agent.chat(user_prompt, max_tokens=10)
        if self.verification.verify(response):
            return response
        simple_fit = self.verification.simple_fit(response)
        if simple_fit is not None:
            return simple_fit
        force_fit = self.verification.force_fit(response)
        return force_fit

    def judge(self, text, max_retry=3):
        if self.debug:
            print("Judge, text:", text)
        self.agent.temperature = self.temperature
        try_id = 0
        result = None

        # Try logits mode first
        while try_id < max_retry:
            result = self.logits_mode(text)
            try_id += 1
            if result is None:
                if self.debug:
                    print("logits mode failed, {} try".format(try_id))
                continue
            else:
                return result
        # Use embedding mode as fallback
        result = self.embedding_mode(text)
        return result

    @classmethod
    def instance(cls, text, max_retry=5, model="gpt-4o", debug=False):
        new_instance = cls(model=model, debug=debug)
        return new_instance.judge(text, max_retry=max_retry)


def nerif(text, model=NERIF_DEFAULT_LLM_MODEL, debug=False):
    return Nerif.instance(text, model=model, debug=debug)


class NerifMatchString:
    def __init__(self, choices: List[str], model=NERIF_DEFAULT_LLM_MODEL, temperature=0):
        self.choices = choices
        self.model = model
        self.prompt = (
            "Given the following text, determine the best choice to make.\n"
            "If it is hard to make the decision, choose the one you think is the most proper.\n"
            "<options>\n"
        )
        index = 0
        for item in self.choices:
            self.prompt += f"{index}. {item}\n"
            index += 1
        self.prompt += "</options>\n"
        self.prompt += "Now the question is:\n"
        self.prompt += "<question>\n"
        self.prompt += (
            "Choose the best choice from the following options.\n" "Only give me the choice ID, only a number: "
        )
        self.temperature = temperature
        self.agent = SimpleChatAgent(
            model=model,
            temperature=temperature,
        )
        self.logits_agent = LogitsAgent(
            model=model,
            temperature=temperature,
        )
        self.verification = NerificationInt(
            possible_values=[x for x in range(0, len(choices))],
        )
        self.instruction_verification = NerificationString(
            possible_values=choices,
        )

    def logits_mode(self, text):
        self.logits_agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>" + text + "</question>"
        user_prompt = self.prompt.rsplit("<question>", 1)
        user_prompt = user_prompt[0] + question + user_prompt[1]
        response = self.logits_agent.chat(user_prompt, max_tokens=1)
        # Fetch the logprobs of the logits
        if response.choices[0].logprobs is None:
            raise Exception("Logprobs not found in the response")
        logprobs = response.choices[0].logprobs["content"][0]
        sorted_logprobs = sorted(logprobs["top_logprobs"], key=lambda x: x["logprob"], reverse=True)
        # Try to find the most likely logprob
        for index in range(len(sorted_logprobs)):
            simple_fit = self.verification.simple_fit(sorted_logprobs[index]["token"])
            if simple_fit is not None:
                return simple_fit
        return None

    def embedding_mode(self, text, max_tokens=300):
        self.agent.temperature = self.temperature
        question = (
            "<question>\n"
            f"{text}"
            "</question>\n"
            "Choose the best route from the following options.\n"
            "Tell me your analysis. Besides ID, I also need your analysis."
        )
        # replace <question> with the text
        user_prompt = self.prompt.rsplit("<question>", 1)
        user_prompt = user_prompt[0] + question + user_prompt[1]
        response = self.agent.chat(user_prompt, max_tokens=max_tokens)
        if self.verification.verify(response):
            return response
        simple_fit = self.verification.simple_fit(response)
        if simple_fit is not None:
            return simple_fit
        force_fit = self.verification.force_fit(response)
        return force_fit

    def match(self, text, max_retry=3):
        self.agent.temperature = self.temperature
        try_id = 0
        while try_id < max_retry:
            result = self.logits_mode(text)
            try_id += 1
            if result is not None:
                return result

        result = self.embedding_mode(text)
        return result

    @classmethod
    def instance(cls, selections, text, max_retry=5, model=NERIF_DEFAULT_LLM_MODEL):
        new_instance = cls(selections, model=model)
        return new_instance.match(text, max_retry=max_retry)


def nerif_match_string(selections, text, model=NERIF_DEFAULT_LLM_MODEL) -> int:
    return NerifMatchString.instance(selections, text, model=model)

def nerif_match(selections, text, model=NERIF_DEFAULT_LLM_MODEL) -> int:
    return NerifMatchString.instance(selections, text, model=model)
