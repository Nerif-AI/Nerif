import json
import os
from typing import Any, Dict, List, Optional, Union

from litellm import completion, embedding
from openai import OpenAI

# OpenAI Models
OPENAI_MODEL: List[str] = [
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-05-13-preview",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-08-06-preview",
    "gpt-4o-2024-09-13",
    "gpt-4o-2024-09-13-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-4-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-2024-04-09-preview",
]
OPENAI_EMBEDDING_MODEL: List[str] = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROXY_URL = os.environ.get("OPENAI_PROXY_URL")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
NERIF_DEFAULT_LLM_MODEL = os.environ.get("NERIF_DEFAULT_LLM_MODEL", "gpt-4o")
NERIF_DEFAULT_EMBEDDING_MODEL = os.environ.get("NERIF_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

def get_litellm_embedding(
    messages: str,
    model: str = NERIF_DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    if model in OPENAI_EMBEDDING_MODEL:
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY
        if base_url is None or base_url == "":
            base_url = OPENAI_API_BASE
    
    response = embedding(
        model=model,
        input=messages,
        api_key=api_key,
        base_url=base_url,
    )

    return response


def get_litellm_response(
    messages: List[Dict[str, str]],
    model: str = NERIF_DEFAULT_LLM_MODEL,
    temperature: float = 0,
    max_tokens: int = 300,
    stream: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    logprobs: bool = False,
    top_logprobs: int = 5,
) -> Any:
    """
    Get a text response from an OpenAI model.

    Parameters:
    - messages (list): The list of messages to send to the model.
    - model (str): Te name of the OpenAI model to use. Default is "gpt-3.5-turbo".
    - temperature (float): The temperature setting for response generation. Default is 0.
    - max_tokens (int): The maximum number of tokens to generate in the response. Default is 300.
    - stream (bool): Whether to stream the response. Default is False.
    - api_key (str): The API key for accessing the OpenAI API. Default is None.
    - batch_size (int): The number of predictions to make in a single request. Default is 1.

    Returns:
    - list: A list of generated text responses if messages is a list, otherwise a single text response.
    """
    if model in OPENAI_MODEL:
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY
        if base_url is None or base_url == "":
            base_url = OPENAI_API_BASE
    # else:
    #     # Current support claude
    #     os.environ["CLAUDE_API_KEY"] = api_key

    if logprobs:
        responses = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    else:
        responses = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

    return responses


def get_ollama_response(
    prompt: Union[str, List[str]],
    url: str = "http://localhost:11434/v1/",
    model: str = "llama3.1",
    max_tokens: int = 300,
    temperature: float = 0,
    stream: bool = False,
    api_key: Optional[str] = "ollama",
) -> Union[str, List[str]]:
    """
    Get a text response from an Ollama model.

    Parameters:
    - prompt (str or list): The input prompt(s) for the model.
    - url (str): The URL of the Ollama API. Default is "http://localhost:11434/api/generate".
    - model (str): The name of the Ollama model. Default is "llama3.1".
    - max_tokens (int): The maximum number of tokens to generate in the response. Default is 300.
    - temperature (float): The temperature setting for response generation. Default is 0.
    - stream (bool): Whether to stream the response. Default is False.
    - api_key (str): The API key for accessing the Ollama API. Default is None.
    - batch_size (int): The number of predictions to make in a single request. Default is 1.

    Returns:
    - str or list: The generated text response(s).
    """
    
    # todo: support batch ollama inference

    response = get_litellm_response(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        api_key=api_key,
        base_url=url,
    )

    return response


class SimpleChatAgent:
    """
    A simple agent class for the Nerif project.
    This class implements a simple chat agent for the Nerif project.
    It uses OpenAI's GPT models to generate responses to user inputs.

    Attributes:
        proxy_url (str): The URL of the proxy server for API requests.
        api_key (str): The API key for authentication.
        model (str): The name of the GPT model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        messages (List[Any]): The conversation history.
        cost_count (dict): Tracks token usage for input and output.

    Methods:
        reset(prompt=None): Resets the conversation history.
        chat(message, append=False, max_tokens=300): Sends a message and gets a response.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
    ):
        # Set the proxy URL and API key
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        # Set the model, temperature, proxy URL, and API key
        self.model = model
        self.temperature = temperature
        self.proxy_url = proxy_url
        self.api_key = api_key

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.cost_count = {"input_token_count": 0, "output_token_count": 0}

    def reset(self, prompt: Optional[str] = None) -> None:
        # Reset the conversation history
        if prompt is None:
            prompt = self.default_prompt

        self.messages: List[Any] = [{"role": "system", "content": prompt}]

    def chat(self, message: str, append: bool = False, max_tokens: int = 300) -> str:
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        if self.model.startswith("ollama"):
            model_name = self.model.split("/")[1]
            result = get_ollama_response(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        elif self.model in OPENAI_MODEL:
            result = get_litellm_response(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")

        text_result = result.choices[0].message.content
        if append:
            self.messages.append({"role": "system", "content": text_result})
        else:
            self.reset()
        return text_result


class SimpleEmbeddingAgent:
    # TODO: support ollama embedding model
    """
    A simple agent for embedding text.

    Attributes:
        proxy_url (str): The URL of the proxy server for API requests.
        api_key (str): The API key for authentication.
        model (str): The name of the embedding model to use.

    Methods:
        encode(string: str) -> List[float]: Encodes a string into an embedding.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
    ):
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key

    def encode(self, string: str) -> List[float]:
        result = get_litellm_embedding(
            messages=string,
            model=self.model,
            api_key=self.api_key
        )

        return result.data[0]["embedding"]


class LogitsAgent:
    # TODO: support ollama logits model
    """
    A simple agent for fetching logits from a model.

    Attributes:
        proxy_url (str): The URL of the proxy server for API requests.
        api_key (str): The API key for authentication.
        model (str): The name of the model to use.
        default_prompt (str): The default system prompt for the chat.
        temperature (float): The temperature setting for response generation.
        messages (List[Any]): The conversation history.
        cost_count (dict): Tracks token usage for input and output.

    Methods:
        chat(message, max_tokens=300, logprobs=True, top_logprobs=5) -> Any:
            Sends a message and gets a response with logits.
    """

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = NERIF_DEFAULT_LLM_MODEL,
        default_prompt: str = "You are a helpful assistant. You can help me by answering my questions.",
        temperature: float = 0.0,
    ):
        if proxy_url is None or proxy_url == "":
            proxy_url = OPENAI_PROXY_URL
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = OPENAI_API_KEY

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key
        self.temperature = temperature

        # Set the default prompt and initialize the conversation history
        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system", "content": default_prompt},
        ]
        self.cost_count = {"input_token_count": 0, "output_token_count": 0}

    def chat(
        self,
        message: str,
        max_tokens: int = 300,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> Any:
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        if self.model in OPENAI_MODEL:
            result = get_litellm_response(
                self.messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")

        return result
