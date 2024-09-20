import os
import inspect
import logging
import builtins
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
import numpy as np
from openai import OpenAI

from nerif.nerif_agent.nerif_agent import (
    LogitsAgent,
    SimpleChatAgent,
    SimpleEmbeddingAgent,
    StructuredAgent,
    OPENAI_MODEL_STRUCTURED
)

logger = logging.getLogger(__name__)

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

class NerifFormat:
    """
    This class is used to convert LLM's output to specific type/format.
    
    """
    
    def __init__(
        self, 
        model: str = NERIF_DEFAULT_LLM_MODEL,
        temperature = 0,
        default_prompt = None,
        debug = False
    ):
        self.debug = debug
        self.simple = True # If the output is a simple variable
        self.model = model
        self.temperature = temperature
        self.builtins = dir(builtins)
        
        self.agent = StructuredAgent(model=model, temperature=temperature)
        self.prompt = (
            "Reply the following question by filling the format I give you.\n"
            "<question>\n"
        )
        if default_prompt is not None:
            self.prompt = default_prompt
        pass

    def wrap_cls(self, cls):
        if not inspect.isclass(cls):
            raise ValueError("The input should be a class.")
        
        if "builtins" in cls.__module__:
            # Built-in class
            self.simple = True
            class RequestWrapper(BaseModel):
                result: cls
        else:
            self.simple = False
            class RequestWrapper(BaseModel, cls):
                pass   
        
        return RequestWrapper

    
    def format_request(self, cls, text: str):
        self.agent.reset()
        if self.debug:
            logger.debug("Formated requrest, text: %s", text)
        response_format = self.wrap_cls(cls)
        self.agent.temperature = self.temperature
        question = "<question>\n" + text + "</question>\n"
        self.prompt = self.prompt.replace("<question>", question)
        response = self.agent.chat(self.prompt, response_format=response_format)
        choice = response_format.model_validate_json(response.choices[0].message.content)
        if self.debug:
            print("Got response: {}".format(response))
        if self.simple:
            return choice.result
        else:
            return choice
        
    
    
    
class NerifVeification:
    """
    This class is used to verify the result of the Nerif.
    If the result is in the possible_value, return the item.
    If the result is not in the possible_value, return None.

    Attributes:
        possible: list[str] = None, model="text-embedding-3-small"
        model: str = "text-embedding-3-small"
        embedding_model: str = "text-embedding-3-small"

    Methods:
        verify(text: str) -> bool:
            Verify if the text is in the possible_value.
        simple_fit(text: str) -> str:
            Find the best fit in the possible_value.
        force_fit(text: str, similarity="cosine") -> str:
            Find the best fit in the possible_value.
    """

    def __init__(
        self,
        possible_value: List[str] = None,
        model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
        value_instruction: List[str] = None,
    ):
        if possible_value == [] or possible_value is None:
            possible_value = ["True", "False"]
        self.original_options = possible_value
        # Convert the possible value to lower case
        self.possible = []
        # (Optional) Additional instructions for each possible value
        self.possible_instruction = []
        # If possible_instruction is not None, record the instruction for each possible value
        for index in range(len(possible_value)):
            self.possible.append(possible_value[index].lower())
            if value_instruction is not None:
                self.possible_instruction.append(value_instruction[index])
            else:
                self.possible_instruction.append("")
        self.embedding = SimpleEmbeddingAgent(model=model)
        self.possible_embed = []
        self.instruction_embed = []
        # Embed the possible value and the possible instruction
        for index in range(len(self.possible)):
            self.possible_embed.append(self.embedding.encode(self.possible[index]))
        if self.possible_instruction is not None:
            for index in range(len(self.possible_instruction)):
                self.instruction_embed.append(
                    self.embedding.encode(self.possible_instruction[index])
                )

    def verify(self, text: str):
        if text.lower() in self.possible:
            return True
        return False

    def simple_fit(self, text: str):
        """
        If there is a possible value in the text, return the original option.
        """
        text = text.lower()
        for index in range(len(self.possible)):
            if self.possible[index].lower() in text:
                return self.original_options[index]
        return None

    def force_fit(self, text: str, similarity="cosine"):
        text_embed = self.embedding.encode(text)
        min_dist = similarity_dist(text_embed, self.possible_embed[0], similarity)
        min_id = 0
        for index in range(1, len(self.possible_embed)):
            dist = similarity_dist(text_embed, self.possible_embed[index], similarity)
            if dist < min_dist:
                min_dist = dist
                min_id = index
        return self.original_options[min_id]
    
    def instruction_fit(self, text: str, similarity="cosine"):
        text_embed = self.embedding.encode(text)
        min_dist = similarity_dist(text_embed, self.instruction_embed[0], similarity)
        min_id = 0
        for index in range(1, len(self.instruction_embed)):
            dist = similarity_dist(text_embed, self.instruction_embed[index], similarity)
            if dist < min_dist:
                min_dist = dist
                min_id = index
        return self.original_options[min_id]


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
        self.agent = SimpleChatAgent(
            model=model, temperature=temperature
        )
        self.logits_agent = LogitsAgent(
            model=model, temperature=temperature
        )
        if self.model in OPENAI_MODEL_STRUCTURED:
            self.structured_request = NerifFormat(model=model, temperature=temperature)
        self.verification = NerifVeification()
        self.debug = debug
        
        if self.debug:
            print("Nerif initialized with model:", model, "temperature:", temperature, "debug:", debug)

    def structured_mode(self, text: str, max_retry=5):
        if self.debug:
            print("Structured mode, text: ", text)
        self.structured_request.temperature = self.temperature
        try_id = 0
        while try_id < max_retry:
            choice = self.structured_request.format_request(bool, text)
            if self.verification.verify(str(choice)):
                # pass verification
                return choice
            self.structured_request.temperature += 0.1
            try_id += 1
        return None
        
        
    def logits_mode(self, text: str):
        if self.debug:
            print("Logits mode, text:", text)
        self.agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.logits_agent.chat(user_prompt, max_tokens=1)
        if self.debug:
            print("Logits mode, response:", response)
        # Fetch the logprobs of the logits
        # TODO: if LLM don't have logprobs, we need to use another method to get the result
        logprobs = response.choices[0].logprobs["content"][0]
        sorted_logprobs = sorted(logprobs["top_logprobs"], key=lambda x: x["logprob"], reverse=True)
        # Try to find the most likely logprob
        for index in range(len(sorted_logprobs)):
            if self.debug:
                print("Logits mode, sorted_logprobs[index]:", sorted_logprobs[index]["token"])
            simple_fit = self.verification.simple_fit(sorted_logprobs[index]["token"])
            if simple_fit is not None:
                if simple_fit == "True":
                    return True
                else:
                    return False
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
            if response == "True":
                return True
            else:
                return False
        simple_fit = self.verification.simple_fit(response)
        if simple_fit is not None:
            if simple_fit == "True":
                return True
            else:
                return False
        force_fit = self.verification.force_fit(response)
        if force_fit == "True":
            return True
        else:
            return False
    
    def judge(self, text, max_retry=3):
        if self.debug:
            print("Judge, text:", text)
        self.agent.temperature = self.temperature
        user_prompt = (
            f"Now the question is:"
            f"<question>\n "
            f"{text}"
            f"</question>\n"
            f"True or False? Remeber, only answer with 'True' or 'False'."
        )
        try_id = 0
        result = ""

        # Try structured mode first
        if self.model in OPENAI_MODEL_STRUCTURED:
            result = self.structured_mode(text)
            if result is not None:
                return result
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


class NerifMatch:
    def __init__(self, requirement, choices, model=NERIF_DEFAULT_LLM_MODEL, temperature=0):
        self.requirement = requirement
        self.choices = choices
        self.model = model
        self.prompt = (
            "Given the following text, determine the best choice to make.\n"
            "If it is hard to make the decision, choose the one you think is the most proper.\n"
            "<options>"
        )
        index = 0
        for item in self.choices:
            index += 1
            self.prompt += f"{index}. {item}\n"
        self.prompt += "</options>"
        self.prompt += (
            "Choose the best choice from the following options.\n"
            "Only give me the choice ID, only a number"
        )
        self.temperature = temperature
        self.agent = SimpleChatAgent(
            model=model, temperature=temperature, default_prompt=self.prompt
        )
        if model in OPENAI_MODEL_STRUCTURED:
            self.structured_request = NerifFormat(model=model, temperature=temperature, default_prompt=self.prompt)
        self.verification = NerifVeification(
            possible_value=[str(x) for x in range(1, index + 1)],
            value_instruction=self.choices
        )

    def id_to_key(self, index):
        return list(self.choice.keys())[index - 1]

    def formated_mode(self, text, max_retry=5):
        self.structured_request.temperature = self.temperature
        try_id = 0
        while try_id < max_retry:
            choice = self.structured_request.format_request(int, text)
            if self.verification.verify(str(choice)):
                # pass verification
                return self.id_to_key(int(choice))
            self.structured_request.temperature += 0.1
            try_id += 1
        return None

        
    def match(self, text, max_retry=5):
        self.agent.temperature = self.temperature
        user_prompt = (
            "<question>\n"
            f"{text}"
            "</question>\n"
            "Choose the best route from the following options.\n"
            "Only give me the choice ID, only a number"
        )
        if self.model in OPENAI_MODEL_STRUCTURED:
            choice = self.formated_mode(text, max_retry=max_retry)
            if choice is not None:
                return choice
        try_id = 0
        choice = ""
        while try_id < max_retry:
            choice = self.agent.chat(user_prompt, max_tokens=50)
            if self.verification.verify(choice):
                # pass verification
                return self.id_to_key(int(choice))
            self.agent.temperature += 0.1
            try_id += 1
        final_prompt = (
            "<question>\n"
            f"{text}"
            "</question>\n"
            "Choose the best route from the following options.\n"
            "Tell me your analysis. Besides ID, I also need your analysis."
        )
        choice = self.agent.chat(final_prompt, max_tokens=300)
        choice = self.complex_verification.force_fit(choice)
        if choice is not None:
            return self.id_to_key(int(choice))
        raise Exception("Failed to verify the result in switch.")

    @classmethod
    def instance(cls, selections, text, max_retry=5, model=NERIF_DEFAULT_LLM_MODEL):
        new_instance = cls(selections, model=model)
        return new_instance.match(text, max_retry=max_retry)


def nerif_match(text, selections, model=NERIF_DEFAULT_LLM_MODEL):
    return NerifMatch.instance(selections, text, model=model)
