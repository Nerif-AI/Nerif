import os
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class nerif_verification:
    def __init__(self, possible_value: list[str] = ["True", "False"]):
        if possible_value == [] or possible_value is None:
            possible_value = ["True", "False"]
        self.possible = [x.lower() for x in possible_value]

    def verify(self, text: str):
        if text.lower() in self.possible:
            return True
        return False
    
    def simple_format(self, text: str):
        text = text.lower()
        for item in self.possible:
            if item in text:
                return item
        return None


class nerif:
    def __init__(self, model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY")):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model
        self.condition = (
            "Given the following text, determine if the statement is true or false.\n"
            "<question>\n"
            "REPLACE_ME"
            "</question>\n"
            "Only answer with 'True' or 'False'."
        )
        self.verification = nerif_verification()

    def judge(self, text, max_retry=5):
        true_prompt = self.condition.replace("REPLACE_ME", text)
        try_id = 0
        while try_id < max_retry:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": true_prompt,
                    }
                ],
                model=self.model,
            )
            result = response.choices[0].message.content
            print(result)
            if self.verification.verify(result):
                if result == "True":
                    return True
                else:
                    return False
            else:
                format_value = self.verification.simple_format(result)
                if format_value is not None:
                    if format_value == "True":
                        return True
                    else:
                        return False
            try_id += 1
        raise Exception("Failed to verify the result in if.")
    
    @classmethod
    def instance(cls, text, max_retry=5, model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY")):
        isinstance = cls(model=model, api_key=api_key)
        return isinstance.judge(text, max_retry=max_retry)
    
        


class nerif_match:
    def __init__(self, choice_dict, model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY")):
        self.choice = choice_dict
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) 
        self.route = (
            "Given the following text, determine the best route to take.\n"
            "Choose the best route from the following options:\n"
        )
        index = 0
        for _, value in self.choice.items():
            index += 1
            self.route += f"{index}: {value}\n"
        self.route += "<question>\n" "REPLACE_ME" "</question>\n" "Only answer with the number."
        self.verification = nerif_verification(possible_value=[str(x) for x in range(1, index + 1)])

    def id_to_key(self, id):
        return list(self.choice.keys())[id - 1]
    
    def match(self, text, max_retry=5):
        true_prompt = self.route.replace("REPLACE_ME", text)
        try_id = 0
        while try_id < max_retry:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": true_prompt,
                    }
                ],
                model=self.model,
            )
            choice = response.choices[0].message.content
            if self.verification.verify(choice):
                # pass verification
                return self.id_to_key(int(choice))
            else:
                format_value = self.verification.simple_format(choice)
                if format_value is not None:
                    return self.id_to_key(int(format_value))
            
            try_id += 1
        raise Exception("Failed to verify the result in switch.")
    
    @classmethod
    def instance(cls, dict, text, max_retry=5, model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY")):
        isinstance = cls(dict, model=model, api_key=api_key)
        return isinstance.match(text, max_retry=max_retry)