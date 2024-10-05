import os
from typing import Optional


class APIStore:
    api_key: str = None
    base_url: str = None
    proxy_url: str = None

    @classmethod
    def postprocess(cls):
        if cls.base_url is not None and cls.base_url[-1] == "/":
            cls.base_url = cls.base_url[:-1]
        if cls.proxy_url is not None and cls.proxy_url[-1] == "/":
            cls.proxy_url = cls.proxy_url[:-1]

    @classmethod
    def setup(cls, api_key: str, base_url: Optional[str], proxy_url: Optional[str]):
        cls.api_key = api_key
        cls.base_url = base_url
        cls.proxy_url = proxy_url
        cls.postprocess()

    @classmethod
    def setup_from_env(cls):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        proxy_url = os.getenv("OPENAI_PROXY_URL")
        cls.setup(api_key=api_key, base_url=base_url, proxy_url=proxy_url)

    @classmethod
    def get_key(cls):
        return cls.api_key

    @classmethod
    def get_base_url(cls):
        return cls.base_url

    @classmethod
    def get_proxy_url(cls):
        return cls.proxy_url

    @classmethod
    def set_env(cls):
        if cls.api_key is not None:
            os.environ["OPENAI_API_KEY"] = cls.api_key
        if cls.base_url is not None:
            os.environ["OPENAI_API_BASE"] = cls.base_url
        if cls.proxy_url is not None:
            os.environ["OPENAI_PROXY_URL"] = cls.proxy_url


class OpenAIAPIStore(APIStore):
    pass
