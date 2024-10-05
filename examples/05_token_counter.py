from nerif.nerif_agent import NerifTokenCounter
from nerif.nerif_core import nerif

if nerif("the sky is blue"):
    print("True")

print(NerifTokenCounter.get_tokens())
