import os

import numpy as np
from openai import OpenAI
from typing import List, Any, Union, Dict, Optional

from nerif.nerif_agent.nerif_agent import SimpleEmbeddingAgent, SimpleChatAgent, LogitsAgent

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")


def similarity_dist(vec1, vec2, func="cosine"):
    if func == "cosine":
        return 1 - (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return np.linalg.norm(vec1 - vec2)


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
        model: str = "text-embedding-3-small",
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
            self.possible.append(value_instruction[index].lower())
            if value_instruction is not None:
                self.possible_instruction.append(value_instruction[index])
            else:
                self.possible_instruction.append("")
        self.embedding = SimpleEmbeddingAgent(model)
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
    def __init__(self, model="gpt-4o", temperature=0):
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
        self.verification = NerifVeification()

    def logits_mode(self, text: str):
        self.agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.logits_agent.chat(user_prompt, max_tokens=1)
        # Fetch the logprobs of the logits
        logprobs = response.logprobs
        sorted_logprobs = sorted(logprobs.logprobs, key=lambda x: x.logprob, reverse=True)
        # Try to find the most likely logprob
        for index in range(len(sorted_logprobs)):
            simple_fit = self.verification.simple_fit(sorted_logprobs[index].token)
            if simple_fit is not None:
                if simple_fit == "True":
                    return True
                else:
                    return False
        return None
    
    def embedding_mode(self, text: str):
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
    
    def judge(self, text, max_retry=3):
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

        while try_id < max_retry:
            result = self.agent.chat(user_prompt, max_tokens=10)
            if self.verification.verify(result):
                if result == "True":
                    return True
                else:
                    return False
            simple_fit = self.verification.simple_fit(result)
            if simple_fit is not None:
                if simple_fit == "True":
                    return True
                else:
                    return False
            try_id += 1
            self.agent.temperature = max(1.0, self.agent.temperature + 0.1)
        result = self.verification.force_fit(result)

        if result == "True":
            return True
        else:
            return False

    @classmethod
    def instance(cls, text, max_retry=5, model="gpt-4o"):
        new_instance = cls(model=model)
        return new_instance.judge(text, max_retry=max_retry)


def nerif(text, model="gpt-4o"):
    return Nerif.instance(text, model=model)


class NerifMatch:
    def __init__(self, choice_dict, model="gpt-4o", temperature=0):
        self.choice = choice_dict
        self.model = model
        self.prompt = (
            "Given the following text, determine the best route to take.\n"
            "If it is hard to make the decision, choose the one you think is the most proper.\n"
            "<options>"
        )
        index = 0
        for _, value in self.choice.items():
            index += 1
            self.prompt += (
                f"<option>"
                f"<id>{index}</id>"
                f"<description>{value}</description>"
                f"</option>\n"
            )
        self.prompt += "</options>"
        self.prompt += (
            "Choose the best route from the following options.\n"
            "Only give me the choice ID, only a number"
        )
        self.temperature = temperature
        self.agent = SimpleChatAgent(
            model=model, temperature=temperature, default_prompt=self.prompt
        )
        self.verification = NerifVeification(
            possible_value=[str(x) for x in range(1, index + 1)]
        )
        self.complex_verification = NerifVeification(
            possible_value=[value for value in self.choice.values()]
        )

    def id_to_key(self, index):
        return list(self.choice.keys())[index - 1]

    def match(self, text, max_retry=5):
        self.agent.temperature = self.temperature
        user_prompt = (
            "<question>\n"
            f"{text}"
            "</question>\n"
            "Choose the best route from the following options.\n"
            "Only give me the choice ID, only a number"
        )
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
    def instance(cls, selections, text, max_retry=5, model="gpt-4o"):
        new_instance = cls(selections, model=model)
        return new_instance.match(text, max_retry=max_retry)


def nerif_match(text, selections, model="gpt-4o"):
    return NerifMatch.instance(selections, text, model=model)
