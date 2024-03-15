from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory


class IChain:
    def create(self, llm: BaseChatModel, memory: ConversationBufferMemory,  *args) -> Chain:
        pass
