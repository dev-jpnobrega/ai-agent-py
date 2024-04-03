from typing import List, Sequence

from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.schema import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage

from ai_enterprise_agent.interface.chat_history import IChatHistoryService


class MemoryChatHistory(BaseChatMessageHistory, IChatHistoryService):

    def __init__(self, limit: int = 5):
        super().__init__()
        self.limit = limit
        self.chat_memory = ChatMessageHistory()
        self.memory = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key='history',
            chat_memory=self.chat_memory,
            k=self.limit
        )

    def get_memory(self):
        return self.memory

    def add_user_message(self, message: str):
        self.memory.chat_memory.add_user_message(message=message)

    def add_ai_message(self, message: str):
        if isinstance(message, AIMessage):
            self.add_message(message)
        else:
            self.add_message(AIMessage(content=message))

    def add_message(self, message: BaseMessage) -> None:
        if type(self).add_messages != BaseChatMessageHistory.add_messages:
            self.add_messages([message])
        else:
            raise NotImplementedError(
                "add_message is not implemented for this class. "
                "Please implement add_message or add_messages."
            )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            self.memory.chat_memory.add_message(message)

    def get_messages(self):
        return self.memory.load_memory_variables

    def clear(self):
        self.memory.clear()


