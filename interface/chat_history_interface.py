from typing import List

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory, BaseMessage


class IChatHistory:
    def addUserMessage(self, message: str) -> None:
        pass

    def addAIChatMessage(self, message: str) -> None:
        pass

    def getMessages(self) -> List[BaseMessage]:
        pass

    def getFormatedMessages(self, messages: List[BaseMessage]) -> str:
        pass

    def clear(self) -> None:
        pass

    def getChatHistory(self) -> BaseChatMessageHistory:
        pass

    def getBufferMemory(self) -> ConversationBufferMemory:
        pass
