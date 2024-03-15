from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory


class IChainService:
    def build(self, llm: BaseChatModel, chat_history: ConversationBufferMemory, *args) -> Chain:
        pass
