from dataclasses import dataclass
from .nerif_agent import OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from typing import List, Dict
import uuid
import tiktoken

def count_tokens_embedding_openai(model_name, messages: str):
  encoding = tiktoken.encoding_for_model(model_name=model_name)
  num_tokens = len(encoding.encode(messages))
  return num_tokens

def count_tokens_embedding(model_name, messages: str):
  if model_name in OPENAI_EMBEDDING_MODEL:
    return count_tokens_embedding_openai(model_name, messages)
  else:
    raise NotImplementedError("Model {} is not supported".format(model_name))

def count_tokens_request_openai(model_name, messages: List[Dict[str, str]]):
  tokens_per_message = 3
  tokens_per_name = 1
  encoding = tiktoken.encoding_for_model(model_name=model_name)
  # num_tokens = 0
  request_tokens = 0
  response_tokens = 0
  for index, message in enumerate(messages):
      # num_tokens += tokens_per_message
      cur_num_token = len(encoding.encode(message["content"]))
      # num_tokens += cur_num_token
      if message["role"] == "system" and index != 0:
        response_tokens += cur_num_token + tokens_per_message
        response_tokens += 1
      elif message["role"] == "user" or (message["role"] and index == 0):
        request_tokens += cur_num_token + tokens_per_message
        request_tokens += 1
      else:
        raise ValueError("Invalid role")
      # print("Message {} -> Token: {}".format(message, cur_num_token))
          # if key == "name":
          #     num_tokens += tokens_per_name
  
  if messages[-1]["role"] == "user":
    request_tokens += 3
  elif messages[-1]["role"] == "system":
    response_tokens -= 4
  request_tokens += 3
  return request_tokens, response_tokens

def count_tokens_request(model_name, messages: List[Dict[str, str]]):
  if model_name in OPENAI_MODEL:
    return count_tokens_request_openai(model_name, messages)
  else:
    raise NotImplementedError("Model {} is not supported".format(model_name))

class NerifTokenConsume:
  def __init__(self, request, response):
    self.request = request
    self.response = response
    self.total = request + response
    
  def __repr__(self) -> str:
    return "Request: {}, Response: {}, Total: {}".format(self.request, self.response, self.total)

class NerifTokenCounter:
  def __init__(self):
    self.model_token = {} 

  def add_message(self, model_name, messages):
    counter_id = str(uuid.uuid4())
    k = (model_name, counter_id)
    if isinstance(messages, str):
      embedding_tokens = count_tokens_embedding(model_name, messages)
      self.model_token[k] = NerifTokenConsume(embedding_tokens, 0)
    elif isinstance(messages, list):
      request_tokens, response_tokens = count_tokens_request(model_name, messages)
      self.model_token[k] = NerifTokenConsume(request_tokens, response_tokens)
    else:
      raise ValueError("Invalid message data structure")

  def add(self, agent):
    model_name = agent.model
    messages = agent.messages
    self.add_message(model_name, messages)