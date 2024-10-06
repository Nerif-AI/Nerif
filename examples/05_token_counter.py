from nerif.agent import NerifTokenCounter
from nerif.core import core

if core("the sky is blue"):
    print("True")

print(NerifTokenCounter.get_tokens())
