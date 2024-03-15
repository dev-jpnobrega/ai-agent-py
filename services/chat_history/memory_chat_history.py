from typing import List

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory, BaseMessage
from langchain.storage.in_memory import InMemoryBaseStore

from interface.agent_interface import IDatabaseConfig
from interface.chat_history_interface import IChatHistory


class MemoryChatHistory(IChatHistory):
    def __init__(self, settings: IDatabaseConfig):
        self._settings = settings
        self._history = BaseChatMessageHistory
        self._bufferMemory = ConversationBufferMemory

    def addUserMessage(self, message: str) -> None:
        if self._history:
            self._history.addUserMessage(message)

    def addAIChatMessage(self, message: str) -> None:
        if self._history:
            self._history.addAIChatMessage(message)

    def getMessages(self) -> List[BaseMessage]:
        messages = self._history.getMessages() if self._history else []
        cut = messages[-(self._settings.get('limit', 5)):]

        return cut

    def getFormatedMessages(self, messages: List[BaseMessage]) -> str:
        formated = '\n'.join(
            f"{message._getType().upper()}: {message.content}" for message in messages
        )

        return formated

    def getChatHistory(self) -> BaseChatMessageHistory:
        return self._history

    def getBufferMemory(self) -> ConversationBufferMemory:
        return self._bufferMemory

    def clear(self) -> None:
        if self._history:
            self._history.clear()

    def build(self) -> IChatHistory:
        self._history = InMemoryBaseStore()

        self._bufferMemory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            chat_memory=self._history
        )

        return self
