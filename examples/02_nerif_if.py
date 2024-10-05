from nerif.nerif_agent import SimpleChatAgent
from nerif.nerif_core import nerif

agent = SimpleChatAgent()

if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    print(agent.chat("what is the color of the sky?"))
