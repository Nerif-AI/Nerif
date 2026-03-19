from typing import Any, List, Optional

from ..model import LogitsChatModel, SimpleChatModel, SimpleEmbeddingModel
from ..utils import (
    LOGGER,
    NERIF_DEFAULT_EMBEDDING_MODEL,
    NERIF_DEFAULT_LLM_MODEL,
    OPENAI_MODEL,
    NerifTokenCounter,
    similarity_dist,
)

# Models known to support the logprobs parameter
_LOGPROBS_SUPPORTED_MODELS = set(OPENAI_MODEL)


def support_logit_mode(model_name):
    # Check exact match
    if model_name in _LOGPROBS_SUPPORTED_MODELS:
        return True
    # Check if it's an OpenRouter-wrapped OpenAI model
    if model_name.startswith("openrouter/openai/"):
        bare = model_name[len("openrouter/openai/") :]
        if bare in _LOGPROBS_SUPPORTED_MODELS:
            return True
    return False


class NerificationBase:
    """
    Base class for Nerification.
    This class is used to verify the result of the Nerif.
    This class provides base functionality for verifying and matching values against a predefined set of possible values.

    Attributes:
        original_options (List[Any]): Original list of possible values before conversion
        possible (List[str]): List of possible values converted to lowercase strings
        embedding (SimpleEmbeddingAgent): Agent used for generating embeddings (lazy init)
        possible_embed (List): List of embeddings for each possible value

    Methods:
        convert(val: Any) -> str:
            Converts a value to lowercase string format

        verify(val: Any) -> bool:
            Checks if a value exists in the possible values list

        simple_fit(val: Any):
            Uses embeddings to find the closest matching possible value

        force_fit(val: Any, similarity="cosine"):
            Uses embeddings to find the closest matching possible value
    """

    def __init__(
        self,
        possible_values: Optional[List[Any]] = None,
        model=NERIF_DEFAULT_EMBEDDING_MODEL,
        counter: Optional[NerifTokenCounter] = None,
    ):
        """
        Initialize the NerificationBase.
        possible_values: list[Any] = None
        model: str = NERIF_DEFAULT_EMBEDDING_MODEL (or None to disable embedding)

        possible_values: list of possible values to verify. Only store lower case string
        """
        self.original_options = possible_values
        # Convert the possible value to lower case
        self.possible = []

        for index in range(len(possible_values)):
            self.possible.append(self.convert(possible_values[index]))

        self.embed_model_name = model
        self._embedding = None  # lazy init
        self._counter = counter
        self.possible_embed = []

    @property
    def embedding(self):
        if self._embedding is None:
            if not self.embed_model_name:
                raise RuntimeError(
                    "Embedding model not configured. Set NERIF_DEFAULT_EMBEDDING_MODEL or pass embed_model parameter."
                )
            self._embedding = SimpleEmbeddingModel(model=self.embed_model_name, counter=self._counter)
        return self._embedding

    @property
    def has_embedding(self) -> bool:
        return bool(self.embed_model_name)

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
        if self.possible_embed == []:
            # Embed the possible value and the possible instruction
            for index in range(len(self.possible)):
                self.possible_embed.append(self.embedding.embed(self.possible[index]))
        text_embed = self.embedding.embed(self.convert(val))
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
        model=NERIF_DEFAULT_EMBEDDING_MODEL,
        counter: Optional[NerifTokenCounter] = None,
    ):
        super().__init__([True, False], model, counter)

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

    def __init__(
        self,
        possible_values: Optional[List[str]] = None,
        model=NERIF_DEFAULT_EMBEDDING_MODEL,
        counter: Optional[NerifTokenCounter] = None,
    ):
        super().__init__(possible_values, model, counter)

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

    def __init__(
        self,
        possible_values: Optional[List[int]] = None,
        model=NERIF_DEFAULT_EMBEDDING_MODEL,
        counter: Optional[NerifTokenCounter] = None,
    ):
        super().__init__(possible_values, model, counter)

    def convert(self, val: Any):
        return str(val)

    def verify(self, val: Any):
        if isinstance(val, int):
            return super().verify(val)
        else:
            return False

    def simple_fit(self, val: Any):
        result = super().simple_fit(val)
        if result is None:
            return None
        return int(result)

    def force_fit(self, val: Any, similarity="cosine"):
        return int(super().force_fit(val, similarity))


class Nerif:
    """
    This class is used to judge the truthfulness of a statement.
    It uses two modes: logits mode and embedding mode.
    Logits mode is faster, but less accurate. Fetch top logprobs from the logits.
    Embedding mode is slower, but more accurate. Embed the text and compare with the possible values.

    Attributes:
        model: str = NERIF_DEFAULT_LLM_MODEL
        embed_model: str = NERIF_DEFAULT_EMBEDDING_MODE
        temperature: float = 0
        counter: Optional[NerifTokenCounter] = None
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

    def __init__(
        self,
        model=NERIF_DEFAULT_LLM_MODEL,
        embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
        temperature=0,
        counter=None,
        debug=False,
    ):
        self.model = model
        self.prompt = (
            "Given the following text, determine if the statement is true or false.\n"
            "<question>\n"
            "Only answer with 'True' or 'False'."
        )
        self.temperature = temperature
        self.agent = SimpleChatModel(model=model, temperature=temperature, counter=counter)
        self.logits_agent = LogitsChatModel(model=model, temperature=temperature, counter=counter)
        self.verification = Nerification(counter=counter, model=embed_model)
        self.debug = debug

        if self.debug:
            LOGGER.debug("Nerif initialized with model: %s, temperature: %s, debug: %s", model, temperature, debug)

    def logits_mode(self, text: str):
        if self.debug:
            LOGGER.debug("Logits mode, text: %s", text)
        self.logits_agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.logits_agent.chat(user_prompt, max_tokens=1)
        if self.debug:
            LOGGER.debug("Logits mode, response: %s", response)
        if not (hasattr(response, "choices") and len(response.choices) > 0):
            return None
        # if choices doesn't have no logprobs, raise an exception
        if not hasattr(response.choices[0], "logprobs") or response.choices[0].logprobs is None:
            return None
        logprobs = response.choices[0].logprobs["content"][0]
        sorted_logprobs = sorted(logprobs["top_logprobs"], key=lambda x: x["logprob"], reverse=True)
        # Try to find the most likely logprob
        for index in range(len(sorted_logprobs)):
            if self.debug:
                LOGGER.debug("Logits mode, sorted_logprobs[index]: %s", sorted_logprobs[index]["token"])
            simple_fit = self.verification.simple_fit(sorted_logprobs[index]["token"])
            if simple_fit is not None:
                return simple_fit
        return None

    def embedding_mode(self, text: str):
        if self.debug:
            LOGGER.debug("Embedding mode, text: %s", text)
        self.agent.temperature = self.temperature
        # replace <question> with the text
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.agent.chat(user_prompt, max_tokens=10)
        try:
            direct_result = int(response)
            if self.verification.verify(direct_result):
                return direct_result
        except (ValueError, TypeError):
            pass
        simple_fit = self.verification.simple_fit(response)
        if simple_fit is not None:
            return simple_fit
        force_fit = self.verification.force_fit(response)
        return force_fit

    def text_fallback_mode(self, text: str):
        """Fallback when no embedding model is available. Uses LLM + string matching only."""
        question = "<question>\n" + text + "</question>\n"
        user_prompt = self.prompt.replace("<question>", question)
        response = self.agent.chat(user_prompt, max_tokens=10)

        # Try simple_fit first (substring match on "true"/"false")
        simple_fit = self.verification.simple_fit(response)
        if simple_fit is not None:
            return simple_fit

        # Try verify (exact match) - convert the response to the original value
        if self.verification.verify(response):
            converted = self.verification.convert(response)
            idx = self.verification.possible.index(converted)
            return self.verification.original_options[idx]

        # Final fallback
        return "true" in response.lower()

    def json_mode(self, text: str):
        """Judge using JSON structured output. Returns True/False or None on failure."""
        import json as _json

        self.agent.temperature = self.temperature
        user_prompt = (
            "Given the following text, determine if the statement is true or false.\n"
            f"<question>\n{text}\n</question>\n"
            'Respond with a JSON object: {"answer": true} or {"answer": false}.'
        )

        try:
            response = self.agent.chat(
                user_prompt,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            parsed = _json.loads(response)
            answer = parsed.get("answer")
            if isinstance(answer, bool):
                return answer
            if isinstance(answer, str):
                if answer.lower() == "true":
                    return True
                if answer.lower() == "false":
                    return False
        except (ValueError, KeyError, TypeError):
            pass
        return None

    def judge(self, text, max_retry=3, strategy=None):
        """Judge truthfulness using a multi-tier chain.

        Args:
            text: Statement to judge.
            max_retry: Max retries for logits mode.
            strategy: List of modes to try in order.
                      Default: ["json", "logits", "embedding", "force_fit"]
        """
        if strategy is None:
            strategy = ["json", "logits", "embedding", "force_fit"]

        if self.debug:
            LOGGER.debug("Judge, text: %s, strategy: %s", text, strategy)
        self.agent.temperature = self.temperature

        for mode in strategy:
            if mode == "json":
                result = self.json_mode(text)
                if result is not None:
                    return result

            elif mode == "logits":
                if support_logit_mode(self.model):
                    for _ in range(max_retry):
                        result = self.logits_mode(text)
                        if result is not None:
                            return result

            elif mode == "embedding":
                if self.verification.has_embedding:
                    return self.embedding_mode(text)
                else:
                    return self.text_fallback_mode(text)

            elif mode == "text_fallback":
                return self.text_fallback_mode(text)

            elif mode == "force_fit":
                if self.verification.has_embedding:
                    return self.verification.force_fit(
                        self.agent.chat(
                            self.prompt.replace(
                                "<question>",
                                "<question>\n" + text + "</question>\n",
                            ),
                            max_tokens=10,
                        )
                    )
                else:
                    return self.text_fallback_mode(text)

        return self.text_fallback_mode(text)

    @classmethod
    def instance(
        cls,
        text,
        max_retry=5,
        model="gpt-4o",
        embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
        debug=False,
        counter=None,
        strategy=None,
    ):
        new_instance = cls(model=model, embed_model=embed_model, debug=debug, counter=counter)
        return new_instance.judge(text, max_retry=max_retry, strategy=strategy)


def nerif(
    text,
    model=NERIF_DEFAULT_LLM_MODEL,
    embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
    debug=False,
    counter=None,
    strategy=None,
):
    return Nerif.instance(text, model=model, embed_model=embed_model, debug=debug, counter=counter, strategy=strategy)


class NerifMatchString:
    """
    This class is used to match the best choice from a list of options.
    It uses two modes: logits mode and embedding mode.
    Logits mode is faster, but less accurate. Fetch top logprobs from the logits.
    Embedding mode is slower, but more accurate. Embed the text and compare with the possible values.

    Attributes:
        choices: List[str]
        model: str = NERIF_DEFAULT_LLM_MODEL
        embed_model: str = NERIF_DEFAULT_EMBEDDING_MODEL
        temperature: float = 0
        counter: Optional[NerifTokenCounter] = None

    Methods:
        logits_mode(text: str) -> int:
            Match the best choice using logits mode.
        embedding_mode(text: str) -> int:
            Match the best choice using embedding mode.
        match(text: str, max_retry: int = 3) -> int:
            Match the best choice using logits mode first, if failed, use embedding mode as fallback.
        instance(choices: List[str], text: str, max_retry: int = 5, model: str = NERIF_DEFAULT_LLM_MODEL, embed_model: str = NERIF_DEFAULT_EMBEDDING_MODEL, debug: bool = False, counter: Optional[NerifTokenCounter] = None) -> int:
            Create a new instance and match the best choice.
    """

    def __init__(
        self,
        choices: List[str],
        model=NERIF_DEFAULT_LLM_MODEL,
        embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
        temperature=0,
        counter=None,
    ):
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
        self.prompt += "Choose the best choice from the following options.\nOnly give me the choice ID, only a number: "
        self.temperature = temperature
        self.agent = SimpleChatModel(model=model, temperature=temperature, counter=counter)
        self.logits_agent = LogitsChatModel(model=model, temperature=temperature, counter=counter)
        self.verification = NerificationInt(
            model=embed_model,
            possible_values=[x for x in range(0, len(choices))],
            counter=counter,
        )
        self.instruction_verification = NerificationString(
            model=embed_model,
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
        if not (hasattr(response, "choices") and len(response.choices) > 0):
            return None
        if not hasattr(response.choices[0], "logprobs") or response.choices[0].logprobs is None:
            return None
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

    def text_fallback_mode(self, text, max_tokens=300):
        """Fallback when no embedding model is available."""
        question = (
            "<question>\n" + text + "</question>\n"
            "Choose the best route from the following options.\n"
            "Tell me your analysis. Besides ID, I also need your analysis."
        )
        user_prompt = self.prompt.rsplit("<question>", 1)
        user_prompt = user_prompt[0] + question + user_prompt[1]
        response = self.agent.chat(user_prompt, max_tokens=max_tokens)

        # Try exact match
        if self.verification.verify(response):
            return response
        # Try simple_fit (substring)
        simple_fit = self.verification.simple_fit(response)
        if simple_fit is not None:
            return simple_fit
        # Final fallback: return 0
        return 0

    def json_mode(self, text: str):
        """Match using JSON structured output. Returns choice index or None on failure."""
        import json as _json

        self.agent.temperature = self.temperature
        options_text = "\n".join(f"{i}. {item}" for i, item in enumerate(self.choices))
        user_prompt = (
            "Given the following options:\n"
            f"<options>\n{options_text}\n</options>\n"
            f"<question>\n{text}\n</question>\n"
            'Choose the best option. Respond with a JSON object: {"choice": N} '
            "where N is the option number (0-indexed)."
        )

        try:
            response = self.agent.chat(
                user_prompt,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            parsed = _json.loads(response)
            choice = parsed.get("choice")
            if isinstance(choice, int) and 0 <= choice < len(self.choices):
                return choice
        except (ValueError, KeyError, TypeError):
            pass
        return None

    def match(self, text, max_retry=3, strategy=None):
        """Match best choice using multi-tier chain."""
        if strategy is None:
            strategy = ["json", "logits", "embedding", "force_fit"]

        self.agent.temperature = self.temperature

        for mode in strategy:
            if mode == "json":
                result = self.json_mode(text)
                if result is not None:
                    return result
            elif mode == "logits":
                if support_logit_mode(self.model):
                    for _ in range(max_retry):
                        result = self.logits_mode(text)
                        if result is not None:
                            return result
            elif mode == "embedding":
                if self.verification.has_embedding:
                    return self.embedding_mode(text)
                else:
                    return self.text_fallback_mode(text)
            elif mode == "text_fallback":
                return self.text_fallback_mode(text)
            elif mode == "force_fit":
                if self.verification.has_embedding:
                    return self.verification.force_fit(
                        self.agent.chat(
                            self.prompt.rsplit("<question>", 1)[0]
                            + "<question>\n"
                            + text
                            + "</question>\n"
                            + self.prompt.rsplit("<question>", 1)[1],
                            max_tokens=300,
                        )
                    )
                else:
                    return 0
        return 0

    @classmethod
    def instance(
        cls,
        selections,
        text,
        max_retry=5,
        model=NERIF_DEFAULT_LLM_MODEL,
        embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
        counter=None,
        strategy=None,
    ):
        new_instance = cls(
            selections,
            model=model,
            embed_model=embed_model,
            counter=counter,
        )
        return new_instance.match(text, max_retry=max_retry, strategy=strategy)


def nerif_match_string(
    selections,
    text,
    model=NERIF_DEFAULT_LLM_MODEL,
    embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
    counter=None,
    strategy=None,
) -> int:
    return NerifMatchString.instance(
        selections,
        text=text,
        model=model,
        embed_model=embed_model,
        counter=counter,
        strategy=strategy,
    )


def nerif_match(
    selections,
    text,
    model=NERIF_DEFAULT_LLM_MODEL,
    embed_model=NERIF_DEFAULT_EMBEDDING_MODEL,
    counter=None,
    strategy=None,
) -> int:
    return NerifMatchString.instance(
        selections,
        text=text,
        model=model,
        embed_model=embed_model,
        counter=counter,
        strategy=strategy,
    )
