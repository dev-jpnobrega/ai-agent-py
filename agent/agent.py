from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStore
# from services.vector_store import VectorStoreFactory
# from langchain.schema import BaseMessage
from nanoid import generate

# from utils.event_helper import EventEmitter
from agent.agent_base import AgentBaseCommand
from interface.agent_interface import (IAgent, IAgentConfig, IDatabaseConfig,
                                       IInputProps, IVectorStoreConfig)
# from helpers.string_helpers import interpolate
from services.chains.chain import ChainService, IChainService
from services.chat_history.chat_history import ChatHistoryFactory, IChatHistory
from services.llm.llm import LLMFactory

EVENTS_NAME = {
    'onMessage': 'onMessage',
    'onToken': 'onToken',
    'onEnd': 'onEnd',
    'onError': 'onError',
    'onMessageSystem': 'onMessageSystem',
    'onMessageHuman': 'onMessageHuman',
}

class Agent(IAgent):
    def __init__(self, settings: IAgentConfig):
        # super().__init__()
        self._name = settings.get('name', 'AssistentAgent')
        self._llm = LLMFactory.create(settings['chat_config'], settings['llm_config'])
        self._chainService = ChainService(settings)
        self._chat_history = None
        self._bufferMemory = ConversationBufferMemory
        # self._logger = console
        self._settings = settings
        # self.setup(settings)

    # def setup(self, settings: IAgentConfig):
    #     if 'vectorStoreConfig' in settings:
    #         self._vectorService = VectorStoreFactory.create(
    #             settings['vectorStoreConfig'], settings['llmConfig']
    #         )

    def buildHistory(self, userSessionId: str, settings: IDatabaseConfig) -> IChatHistory:
        if self._chat_history:
            return self._chat_history

        customSettings = settings if settings else {}
        self._chat_history = ChatHistoryFactory.create(
            {**customSettings, 'sessionId': userSessionId or generate()}
        )
        return self._chat_history

    # async def buildRelevantDocs(self, args: IInputProps, settings: IVectorStoreConfig) -> dict:
    #     if not settings:
    #         return {'relevantDocs': [], 'referenciesDocs': []}

    #     customFilters = settings.get('customFilters', None)
    #     vectorFields = settings.get('vectorFieldName', None)

    #     filter = interpolate(customFilters, args) if customFilters else ''

    #     relevantDocs = await self._vectorService.similaritySearch(
    #         args['question'], 10, {'vectorFields': vectorFields, 'filter': filter}
    #     )

    #     referenciesDocs = ', '.join([doc['metadata'] for doc in relevantDocs])

    #     return {'relevantDocs': relevantDocs, 'referenciesDocs': referenciesDocs}

    def call(self, args: IInputProps) -> None:
        question, chatThreadID = args['question'], args['chatThreadID']
        try:
            chat_history = self.buildHistory(chatThreadID, self._settings.get('db_history_config'))
            # relevantDocs = await self.buildRelevantDocs(args, self._settings.get('vectorStoreConfig', {}))
            memory = chat_history.getBufferMemory()
            chain = self._chainService.build(self._llm, memory, question)

            chatMessages = chat_history.getMessages()

            result = chain.invoke(
                input={
                    # 'referencies': relevantDocs['referenciesDocs'],
                    # # 'input_documents': relevantDocs['relevantDocs'],
                    'question': question,
                    'query': question,
                    'chat_history': chatMessages,
                    'format_chat_messages': chat_history.getFormatedMessages(chatMessages),
                    # 'user_prompt': self._settings.get('systemMesssage', ''),
                }
            )
            return result

            # await chatHistory.addUserMessage(question)
            # await chatHistory.addAIChatMessage(result.get('text', ''))

            # self.emit(EVENTS_NAME['onMessage'], result.get('text', ''))
            # self.emit(EVENTS_NAME['onEnd'], 'terminated')

        except Exception as e:
            print(e)
            # self.emit(EVENTS_NAME['onError'], e)

    def execute(self, args: any) -> None:
        raise RuntimeError(args)
