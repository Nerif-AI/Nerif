import json
import os
from typing import List, Any
from litellm import completion
import requests
from openai import OpenAI

# OpenAI Models
# From: https://platform.openai.com/docs/models/gpt-4o
OPENAI_MODEL = [
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


def get_ollama_response(
    prompt,
    url="http://localhost:11434/api/generate",
    model="llama3.1",
    temperature=0,
    stream=False,
    api_key=None,
    batch_size=1,
):
    """
    Get a text response from an Ollama model.

    Parameters:
    - prompt (str or list): The input prompt(s) for the model.
    - url (str): The URL of the Ollama API. Default is "http://localhost:11434/api/generate".
    - model (str): The name of the Ollama model. Default is "llama3.1".
    - temperature (float): The temperature setting for response generation. Default is 0.
    - stream (bool): Whether to stream the response. Default is False.
    - api_key (str): The API key for accessing the Ollama API. Default is None.
    - batch_size (int): The number of predictions to make in a single request. Default is 1.

    Returns:
    - str or list: The generated text response(s).
    """

    if isinstance(prompt, list):
        responses = []
        for p in prompt:
            payload = {
                "model": model,
                "prompt": p,
                "options": {"temperature": temperature, "num_predict": batch_size},
                "stream": stream,
            }

            headers = {"Content-Type": "application/json"}

            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.post(url, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                response_data = response.json()
                responses.append(response_data.get("response", ""))
            else:
                raise Exception(
                    f"Request failed with status code {response.status_code}"
                )

        return responses
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": batch_size},
            "stream": stream,
        }

        headers = {"Content-Type": "application/json"}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("response", "")
        else:
            raise Exception(f"Request failed with status code {response.status_code}")


def get_litellm_response(
    messages,
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=300,
    stream=False,
    api_key=None,
    base_url=None,
    logprobs=False,
    top_logprobs=5,
):
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
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None or base_url == "":
            base_url = os.environ.get("OPENAI_API_BASE")
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = base_url
    else:
        # Current support claude
        os.environ["CLAUDE_API_KEY"] = api_key

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
        proxy_url=None,
        api_key=None,
        model="gpt-3.5-turbo",
        default_prompt="You are a helpful assistant. You can help me by answering my questions.",
        temperature=0.0,
    ):
        # Set the proxy URL and API key
        if proxy_url is None or proxy_url == "":
            proxy_url = os.environ.get("OPENAI_PROXY_URL")
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = os.environ.get("OPENAI_API_KEY")

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

    def reset(self, prompt=None):
        # Reset the conversation history
        if prompt is None:
            prompt = self.default_prompt

        self.messages: List[Any] = [{"role": "system", "content": prompt}]

    def chat(self, message, append=False, max_tokens=300):
        # Append the user's message to the conversation history
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)

        if self.model.startswith("ollama"):
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
    def __init__(self, proxy_url=None, api_key=None, model="text-embedding-3-small"):
        if proxy_url is None or proxy_url == "":
            proxy_url = os.environ.get("OPENAI_PROXY_URL")
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = os.environ.get("OPENAI_API_KEY")

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key

    def encode(self, string):
        if self.proxy_url is None or self.proxy_url == "":
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=string, model=self.model)
            result = response.data[0].embedding
        else:
            payload = json.dumps({"model": self.model, "input": string})
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
            }
            response = requests.request(
                "POST", f"{self.proxy_url}/v1/embeddings", headers=headers, data=payload
            )
            if response.status_code != 200:
                raise Exception(f"Failed to call the proxy server: {response.text}")
            else:
                response_text = response.text
                result_json = json.loads(response_text)
                result = result_json["data"][0]["embedding"]

        return result


class LogitsAgent:
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
    """

    def __init__(
        self,
        proxy_url=None,
        api_key=None,
        model="gpt-3.5-turbo",
        default_prompt="You are a helpful assistant. You can help me by answering my questions.",
        temperature=0.0,
    ):
        if proxy_url is None or proxy_url == "":
            proxy_url = os.environ.get("OPENAI_PROXY_URL")
        elif proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]
        if api_key is None or api_key == "":
            api_key = os.environ.get("OPENAI_API_KEY")

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
        self, message, append=False, max_tokens=300, logprobs=True, top_logprobs=5
    ):
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
