from prettytable import PrettyTable


class ModelCost:
    def __init__(self, model_name, request=0, response=0):
        self.model_name = model_name
        self.request = request
        self.response = response

    def add_cost(self, request, response=0):
        self.request += request
        self.response += response

    def __repr__(self) -> str:
        return f"{self.request} tokens requested, {self.response} tokens returned"


class NerifTokenConsume:
    def __init__(self):
        self.model_cost = {}

    def __getitem__(self, key):
        return self.model_cost[key]

    def append(self, consume: ModelCost):
        if consume is not None:
            model_name = consume.model_name
            if self.model_cost.get(model_name) is None:
                self.model_cost[model_name] = ModelCost(model_name)
            self.model_cost[model_name].add_cost(consume.request, consume.response)

        return self

    def __repr__(self) -> str:
        table = PrettyTable()
        table.field_names = [
            "model name",
            "requested tokens",
            "response tokens",
        ]
        for key, value in self.model_cost.items():
            table.add_row([key, value.request, value.response])
        return table.get_string()


class ResponseParserBase:
    def __call__(self, response) -> NerifTokenConsume:
        raise NotImplementedError("ResponseParserBase __call__ is not implemented")


class OpenAIResponseParser(ResponseParserBase):
    def __call__(self, response) -> ModelCost:
        model_name = response.model
        response_type = response.__class__.__name__
        if response_type == "EmbeddingResponse":
            requested_tokens = len(response.data[0]["embedding"])
            completation_tokens = 0
        else:
            usage = response.usage
            requested_tokens = usage.prompt_tokens
            completation_tokens = usage.completion_tokens

        consume = ModelCost(model_name, requested_tokens, completation_tokens)
        return consume


class OllamaResponseParser(ResponseParserBase):
    def __call__(self, response) -> ModelCost:
        model_name = response.model
        requested_tokens = response.prompt_eval_count
        completation_tokens = response.eval_count

        consume = ModelCost(model_name, requested_tokens, completation_tokens)
        return consume


class NerifTokenCounter:
    """
    Class for counting tokens consumed by the model
    members:
    - model_token: Dict[(str, uuid.UUID), NerifTokenConsume] - dictionary for storing token consumption
    """

    def __init__(self, response_parser: ResponseParserBase = OpenAIResponseParser()):
        """
        Class for counting tokens consumed by the model
        """
        self.model_token = NerifTokenConsume()
        self.response_parser = response_parser

    def set_parser(self, parser: ResponseParserBase):
        self.response_parser = parser

    def set_parser_based_on_model(self, model_name: str):
        if model_name.startswith("ollama"):
            self.set_parser(OllamaResponseParser())
        else:
            self.set_parser(OpenAIResponseParser())

    def count_from_response(self, response):
        """
        Counting tokens consumed by the model from response

        paramaters:
        - response: any - response from the model
        """
        consume = self.response_parser(response)
        self.model_token.append(consume)
