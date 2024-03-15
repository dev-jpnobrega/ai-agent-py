from typing import List

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory, BaseMessage

from interface.agent_interface import IDatabaseConfig
from interface.chat_history_interface import IChatHistory


class RedisChatHistory(IChatHistory):
    def __init__(self, settings: IDatabaseConfig):
        self._settings = settings
        self._redisClientInstance = None
        self._history = None
        self._bufferMemory = None

    def createClient(self):
        if self._redisClientInstance:
            return self._redisClientInstance

        from redis import Redis

        client = Redis(
            **self._settings,
            db=int(self._settings['database']),
            tls={}
        )

        self._redisClientInstance = client
        return self._redisClientInstance

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

