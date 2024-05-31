from nerif import nerif_match

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

from openai import OpenAI
import os
file_content = open(__file__).read()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

condition = "What's this program doing?\n" + "```python\n" + file_content + "```"

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": condition,
        }
    ],
    model="gpt-3.5-turbo",
)
print(response.choices[0].message.content)