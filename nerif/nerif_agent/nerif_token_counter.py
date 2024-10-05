from dataclasses import dataclass


@dataclass
class NerifTokenCounter:
  request_token = 0
  response_token = 0
  embedding_token = 0
  
  @classmethod
  def count_tokens_request(cls, request_count: int):
    cls.request_token += request_count
  
  @classmethod
  def count_tokens_response(cls, response_count: int):
    cls.response_token += response_count
    
  @classmethod
  def count_tokens_embedding(cls, embedding_count: int):
    cls.embedding_token += embedding_count 
   
  @classmethod 
  def reset(cls):
    cls.request_token = 0
    cls.response_token = 0
  
  @classmethod
  def get_tokens(cls):
    return cls.request_token, cls.response_token, cls.embedding_token