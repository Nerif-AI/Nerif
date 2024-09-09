import json
import os
from typing import List, Any

import requests
from openai import OpenAI


class SimpleChatAgent:
    """
    A simple agent class for the Nerif project.
    Parameters:
    - proxy_url (str): The URL of the proxy server. Default is an empty string.
    - model (str): The name of the language model to use. Default is "gpt-4o".
    - api_key (str): The API key for accessing the OpenAI API. Default is the value of the "OPENAI_API_KEY"
    environment variable.
    Methods:
    - __init__(self, proxy_url="", model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY")): Initializes the
    Nerif agent.
    - call(self, messages, model=None): Sends a list of messages to the language model and returns the generated
    response.
    """

    def __init__(self,
                 proxy_url=None,
                 api_key=None,
                 model="gpt-4o",
                 default_prompt="You are a helpful assistant. You can help me by answering my questions.",
                 temperature=0.1):
        if proxy_url is None or proxy_url == "":
            proxy_url = os.environ.get("OPENAI_PROXY_URL")
        if api_key is None or api_key == "":
            api_key = os.environ.get("OPENAI_API_KEY")
        if proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]

        self.model = model
        self.temperature = temperature
        self.proxy_url = proxy_url
        self.api_key = api_key

        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system",
             "content": default_prompt},
        ]
        self.cost_count = {"input_token_count": 0, "output_token_count": 0}

    def reset(self, prompt=None):
        if prompt is None:
            prompt = self.default_prompt

        self.messages: List[Any] = [
            {"role": "system",
             "content": prompt}
        ]

    def chat(self, message, append=False, max_tokens=300):
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)
        # print(self.messages)
        if self.proxy_url is None or self.proxy_url == "":
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            stat = response.usage
            self.cost_count["input_token_count"] += stat.prompt_tokens
            self.cost_count["output_token_count"] += stat.completion_tokens
        else:
            payload = json.dumps({
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens
            })
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", f"{self.proxy_url}/v1/chat/completions", headers=headers, data=payload)
            if response.status_code != 200:
                raise Exception(f"Failed to call the proxy server: {response.text}")
            else:
                response_text = response.text
                result_json = json.loads(response_text)
                result = result_json["choices"][0]["message"]["content"]
                stat = result_json["usage"]
                self.cost_count["input_token_count"] += stat["prompt_tokens"]
                self.cost_count["output_token_count"] += stat["completion_tokens"]

        if append:
            self.messages.append({"role": "system", "content": result})
        else:
            self.reset()
        return result


class SimpleEmbeddingAgent:
    def __init__(self,
                 proxy_url=None,
                 api_key=None,
                 model="text-embedding-3-small"):
        if proxy_url is None or proxy_url == "":
            proxy_url = os.environ.get("OPENAI_PROXY_URL")
        if api_key is None or api_key == "":
            api_key = os.environ.get("OPENAI_API_KEY")
        if proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]

        self.model = model
        self.proxy_url = proxy_url
        self.api_key = api_key

    def encode(self, string):
        if self.proxy_url is None or self.proxy_url == "":
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                input=string,
                model=self.model
            )
            result = response.data[0].embedding
        else:
            payload = json.dumps({
                "model": self.model,
                "input": string
            })
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", f"{self.proxy_url}/v1/embeddings", headers=headers, data=payload)
            if response.status_code != 200:
                raise Exception(f"Failed to call the proxy server: {response.text}")
            else:
                response_text = response.text
                result_json = json.loads(response_text)
                result = result_json["data"][0]["embedding"]

        return result


class LogitsAgent:
    """
    Get Logits from OpenAI API
    Parameters:
    - proxy_url (str): The URL of the proxy server. Default is an empty string.
    - model (str): The name of the language model to use. Default is "gpt-4o".
    - api_key (str): The API key for accessing the OpenAI API. Default is the value of the "OPENAI_API_KEY"
    environment variable.
    Methods:
    - __init__(self, proxy_url="", model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY")): Initializes the
    Nerif agent.
    - call(self, messages, model=None): Sends a list of messages to the language model and returns the generated
    response.
    """

    def __init__(self,
                 proxy_url=None,
                 api_key=None,
                 model="gpt-4o",
                 default_prompt="You are a helpful assistant. You can help me by answering my questions.",
                 temperature=0.0):
        if proxy_url is None or proxy_url == "":
            proxy_url = os.environ.get("OPENAI_PROXY_URL")
        if api_key is None or api_key == "":
            api_key = os.environ.get("OPENAI_API_KEY")
        if proxy_url[-1] == "/":
            proxy_url = proxy_url[:-1]

        self.model = model
        self.temperature = temperature
        self.proxy_url = proxy_url
        self.api_key = api_key

        self.default_prompt = default_prompt
        self.messages: List[Any] = [
            {"role": "system",
             "content": default_prompt},
        ]
        self.cost_count = {"input_token_count": 0, "output_token_count": 0}

    def reset(self, prompt=None):
        if prompt is None:
            prompt = self.default_prompt

        self.messages: List[Any] = [
            {"role": "system",
             "content": prompt}
        ]

    def chat(self, message, append=False, max_tokens=300):
        new_message = {"role": "user", "content": message}
        self.messages.append(new_message)
        # print(self.messages)
        if self.proxy_url is None or self.proxy_url == "":
            client = OpenAI(api_key=self.api_key)
            response = client.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=2,
            )
            result = response.choices[0].message.content
            stat = response.usage
            self.cost_count["input_token_count"] += stat.prompt_tokens
            self.cost_count["output_token_count"] += stat.completion_tokens
        else:
            payload = json.dumps({
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "logprobs": True,
                "top_logprobs": 2,
            })
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
            }
            print("!!!!")
            print(payload)
            print("!!!!")
            response = requests.request("POST", f"{self.proxy_url}/v1/chat/completions", headers=headers, data=payload)
            if response.status_code != 200:
                raise Exception(f"Failed to call the proxy server: {response.text}")
            else:
                response_text = response.text
                result_json = json.loads(response_text)
                result = result_json["choices"][0]["message"]["content"]
                stat = result_json["usage"]
                self.cost_count["input_token_count"] += stat["prompt_tokens"]
                self.cost_count["output_token_count"] += stat["completion_tokens"]

        return response
        if append:
            self.messages.append({"role": "system", "content": result})
        else:
            self.reset()
        return result

