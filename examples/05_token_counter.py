from nerif.nerif_core import nerif
from nerif.nerif_agent import NerifTokenCounter

if nerif("the sky is blue"):
    print("True")

print(NerifTokenCounter.get_tokens())