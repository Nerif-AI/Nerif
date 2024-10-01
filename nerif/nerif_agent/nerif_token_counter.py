from dataclasses import dataclass
from .utils import OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
from typing import List, Dict
import uuid
import tiktoken

def count_tokens_embedding_openai(model_name, messages: str):
  """
  Util function for counting tokens for OpenAI embedding model
  
  paramaters:
  - model_name: str - model name
  - messages: str - messages to be encoded

  return:
  - num_tokens: int - number of tokens
  """
  encoding = tiktoken.encoding_for_model(model_name=model_name)
  num_tokens = len(encoding.encode(messages))
  return num_tokens

def count_tokens_embedding(model_name, messages: str):
  """
  Util function for routing to the correct function for counting tokens
  
  paramaters:
  - model_name: str - model name
  - messages: str - messages to be encoded
  
  return:
  - num_tokens: int - number of tokens
  
  raise:
  - NotImplementedError: if model provider is not supported
  """
  if model_name in OPENAI_EMBEDDING_MODEL:
    return count_tokens_embedding_openai(model_name, messages)
  else:
    raise NotImplementedError("Model {} is not supported".format(model_name))

def count_tokens_request_openai(model_name, messages: List[Dict[str, str]]):
  """
  Util function for counting tokens for OpenAI model
  
  paramaters:
  - model_name: str - model name
  - messages: List[Dict[str, str]] - all messages from the chat
    [
      {"role": "user", "content": "Hello"},
      {"role": "system", "content": "Hi"}
      ....
    ]
  
  return:
  - request_tokens: int - number of tokens for request
  - response_tokens: int - number of tokens for response
  
  raise:
  - ValueError: if the role is not user or system
  """
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
  """
  Util function for routing to the correct function for counting tokens
  
  paramaters:
  - model_name: str - model name
  - messages: List[Dict[str, str]] - all messages from the chat
    [
      {"role": "user", "content": "Hello"},
      {"role": "system", "content": "Hi"}
      ....
    ]
    
  return:
  - request_tokens: int - number of tokens for request
  - response_tokens: int - number of tokens for response
  
  raise:
  - NotImplementedError: if model provider is not supported
  """
  if model_name in OPENAI_MODEL:
    return count_tokens_request_openai(model_name, messages)
  else:
    raise NotImplementedError("Model {} is not supported".format(model_name))

class NerifTokenConsume:
  """
  Data structure for storing token consumption
  members:
  - request: int - number of tokens for request
  - response: int - number of tokens for response
  - total: int - total number of tokens consumed
  """
  def __init__(self, request, response):
    """
    Data structure for storing token consumption
    
    paramaters:
    - request: int - number of tokens for request
    - response: int - number of tokens for response
    """
    self.request = request
    self.response = response
    self.total = request + response
    
  def __repr__(self) -> str:
    """
    String representation of the object
    
    return:
    - str: string representation
    """
    return "Request: {}, Response: {}, Total: {}".format(self.request, self.response, self.total)
  
  def append(self, consume):
    """
    Add another consume to the current object
    self.request += consume.request
    self.response += consume.response
    
    paramaters:
    - consume: NerifTokenConsume - object to be appended
    
    return:
    - self: NerifTokenConsume - self object
    """
    if consume is None:
      return self
    self.request += consume.request
    self.response += consume.response
    self.total += consume.total
    return self

class NerifTokenCounter:
  """
  Class for counting tokens consumed by the model
  members:
  - model_token: Dict[(str, uuid.UUID), NerifTokenConsume] - dictionary for storing token consumption
  """
  def __init__(self):
    """
    Class for counting tokens consumed by the model
    """
    self.model_token = {} 

  def add_message(self, model_name, messages, chat_id=None):
    """
    Counting tokens consumed by the model in specific chat
    
    paramaters:
    - model_name: str - model name
    - messages: str or List[Dict[str, str]] - messages to be added
    - chat_id: uuid.UUID - chat id (Optional)
      - if None, generate new uuid
    
    raise:
    - ValueError: if message data structure is invalid
    """
    if chat_id is None:
      chat_id = uuid.uuid4()
    k = (model_name, chat_id)
    consume = self.model_token.get(k)
    if isinstance(messages, str):
      embedding_tokens = count_tokens_embedding(model_name, messages)
      self.model_token[k] = NerifTokenConsume(embedding_tokens, 0).append(consume)
    elif isinstance(messages, list):
      request_tokens, response_tokens = count_tokens_request(model_name, messages)
      self.model_token[k] = NerifTokenConsume(request_tokens, response_tokens).append(consume)
    else:
      raise ValueError("Invalid message data structure")

  def add(self, agent):
    """
    Count tokens consumed by the agent in current state
    
    paramaters:
    - agent: NerifAgent - agent to be added
    """
    model_name = agent.model
    messages = agent.messages
    chat_id = agent.chat_uuid
    self.add_message(model_name, messages, chat_id=chat_id)