import os

import numpy as np
from openai import OpenAI

from nerif.nerif_agent.nerif_agent import SimpleEmbeddingAgent, SimpleChatAgent

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")


def similarity_dist(vec1, vec2, func="cosine"):
    if func == "cosine":
        return 1 - (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        return np.linalg.norm(vec1 - vec2)


class NerifVeification:
    def __init__(self, possible_value: list[str] = None, model="text-embedding-3-small"):
        if possible_value is None:
            possible_value = ["True", "False"]
        if possible_value == [] or possible_value is None:
            possible_value = ["True", "False"]
        self.possible = [x.lower() for x in possible_value]
        self.embedding = SimpleEmbeddingAgent(model)
        self.possible_embed = []
        for index in range(len(self.possible)):
            self.possible_embed.append(self.embedding.encode(self.possible[index]))

    def verify(self, text: str):
        if text.lower() in self.possible:
            return True
        return False

    def simple_fit(self, text: str):
        text = text.lower()
        for item in self.possible:
            if item in text:
                return item
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
        return self.possible[min_id]


class Nerif:
    def __init__(self, model="gpt-4o", temperature=0):
        self.model = model
        self.prompt = (
            "Given the following text, determine if the statement is true or false.\n"
            "Only answer with 'True' or 'False'."
        )
        self.temperature = temperature
        self.agent = SimpleChatAgent(model=model, temperature=temperature,
                                     default_prompt=self.prompt)
        self.verification = NerifVeification()

    def judge(self, text, max_retry=5):
        self.agent.temperature = self.temperature
        user_prompt = (f"Now the question is:"
                       f"<question>\n "
                       f"{text}"
                       f"</question>\n"
                       f"True or False? (Remeber, only) answer with 'True' or 'False'.")
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
            self.prompt += (f"<option>"
                            f"<id>{index}</id>"
                            f"<description>{value}</description>"
                            f"</option>\n")
        self.prompt += "</options>"
        self.prompt += "Choose the best route from the following options.\n" "Only give me the choice ID, only a number"
        self.temperature = temperature
        self.agent = SimpleChatAgent(model=model, temperature=temperature,
                                     default_prompt=self.prompt)
        self.verification = NerifVeification(possible_value=[str(x) for x in range(1, index + 1)])
        self.complex_verification = NerifVeification(possible_value=[value for value in self.choice.values()])

    def id_to_key(self, index):
        return list(self.choice.keys())[index - 1]

    def match(self, text, max_retry=5):
        self.agent.temperature = self.temperature
        user_prompt = "<question>\n" f"{text}" "</question>\n" "Choose the best route from the following options.\n" \
                      "Only give me the choice ID, only a number"
        try_id = 0
        choice = ""
        while try_id < max_retry:
            choice = self.agent.chat(user_prompt, max_tokens=50)
            if self.verification.verify(choice):
                # pass verification
                return self.id_to_key(int(choice))
            self.agent.temperature += 0.1
            try_id += 1
        final_prompt = "<question>\n" f"{text}" "</question>\n" "Choose the best route from the following options.\n" \
                       "Tell me your analysis. Besides ID, I also need your analysis."
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
