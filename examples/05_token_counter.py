from nerif.core import nerif
from nerif.model import NerifTokenCounter

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

print(counter.model_token)
