from nerif.nerif_core import nerif
from nerif.nerif_agent import SimpleChatAgent

agent = SimpleChatAgent()

if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    print(agent.chat("what is the color of the sky?"))