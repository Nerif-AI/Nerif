from nerif.nerif_core import nerif
from nerif.nerif_agent import NerifTokenCounter, SimpleChatAgent
import os
import litellm


counter = NerifTokenCounter()
counter.add_message("text-embedding-3-small", "the sky is blue")
print(counter.model_token)


agent = SimpleChatAgent()
agent_counter = NerifTokenCounter()
print(agent.chat("What is the capital of the moon?"))
print(agent.chat("What is the capital of the moon in the world of Gundam?"))
print(agent.chat("Which mobile suit is the MVP in the Gundam U.C. series", append=True))
print(agent.counter.model_token)


external_counter = NerifTokenCounter()
agent1 = SimpleChatAgent(counter=external_counter)
print(agent1.chat("What is the capital of the moon?", max_tokens=10))
agent2 = SimpleChatAgent(counter=external_counter)
print(agent2.chat("What is the top of the earth", max_tokens=10, append=True))
print(external_counter.model_token)

external_counter = NerifTokenCounter()
judge = nerif("the sky is blue", counter=external_counter)
print(external_counter.model_token)