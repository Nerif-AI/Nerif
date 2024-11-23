from nerif.core import nerif
from nerif.utils import NerifTokenCounter

counter = NerifTokenCounter()

if nerif("the sky is blue", counter=counter):
    print("True")

print(counter.model_token)
