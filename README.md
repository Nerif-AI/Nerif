# Nerif

LLM-powered Python, make you write Python code with natural language.

Currently Nerif supports and will support following models/providers:

- [x] OpenAI  
- [ ] ...  
- [ ] ...


## How to install

```
pip install Nerif
```

## Setup

Before you using Nerif, you should setup LLM model.

### OpenAI

To use OpenAI GPT as your backend, set this environment variable

```
OPENAI_API_KEY=<YOUR API KEY>
```

## Example Usage

> You can find all examples under `test`

Build a `if` statement with LLM

```python
from nerif import nerif

if nerif.instance("Is the sky green?"):
    print("this is true")
else:
    print("this is false")
```

Build a `match` statement with LLM

```python
def func1():
    print("creating new user ...")

def func2():
    print("deleting user ...")

def func3():
    print("making reservation ...")

choice_dict = {
    "func1": "I can create new user in this function",
    "func2": "I can delete user in this function",
    "func3": "I can make reservation for user in this function"
}

match nerif_match.instance(choice_dict, "I wanna use gala server GPU 1-4 tonight"):
    case "func1":
        func1()
    case "func2":
        func2()
    case "func3":
        func3()
```
