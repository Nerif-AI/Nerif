from nerif.agent import NerifTokenCounter
from nerif.core import nerif

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

print(counter.model_token)
