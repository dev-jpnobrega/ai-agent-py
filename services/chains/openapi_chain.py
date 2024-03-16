# from langchain.chains import BaseChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory

from interface.agent_interface import IOpenAPIConfig

from .chain import IChain
from .openapi_base_chain import OpenApiBaseChain


class OpenAPIChain(IChain):
    def __init__(self, settings: IOpenAPIConfig) -> None:
        self._settings = settings

    def getHeaders(self) -> dict:
        headers = {}
        if self._settings.get("xApiKey"):
            headers["x-api-key"] = self._settings["xApiKey"]
        if self._settings.get("authorization"):
            headers["Authorization"] = self._settings["authorization"]
        return headers or None

    def create(self, llm: BaseChatModel, *args):
        headers = self.getHeaders()
        open_api_chain = OpenApiBaseChain(
            input={
                'llm': llm,
                'spec': self._settings["data"],
                'customize_system_message': self._settings.get("custom_system_message", ""),
                'headers': headers,
            },
        )
        return open_api_chain
