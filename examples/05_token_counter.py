from nerif.nerif_core import nerif
from nerif.nerif_agent import NerifTokenCounter, SimpleChatAgent
import os
import litellm

# os.environ["LITELLM_LOG"] = "DEBUG"

litellm.set_verbose=True
# if nerif("the sky is blue"):
#     print("True")

counter = NerifTokenCounter()
counter.add_message("text-embedding-3-small", "the sky is blue")
print(counter.model_token)

agent = SimpleChatAgent()
agent_counter = NerifTokenCounter()

print(agent.chat("What is the capital of the moon?"))
agent_counter.add(agent)
print(agent_counter.model_token)

print(agent.chat("What is the capital of the moon?", max_tokens=10))
print(agent.chat("What is the top of the earth", max_tokens=10, append=True))
agent_counter.add(agent)
print(agent_counter.model_token)
