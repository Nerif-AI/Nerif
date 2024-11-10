from nerif.agent import SimpleChatAgent
from nerif.core import nerif, Nerif

agent = SimpleChatAgent()

if nerif("the sky is blue"):
    print("True")
else:
    print("No", end=", ")
    print(agent.chat("what is the color of the sky?"))


judger = Nerif(model="gpt-4o-mini")
print(judger.judge("the sky is blue"))