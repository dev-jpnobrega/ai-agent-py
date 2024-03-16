import asyncio
from pprint import pprint
from typing import List

# from .sql_chain import SqlChain
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (BasePromptTemplate, ChatPromptTemplate,
                               HumanMessagePromptTemplate, MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from interface.agent_interface import SYSTEM_MESSAGE_DEFAULT, IAgentConfig
from interface.chain_interface import IChain
from interface.chain_service_interface import IChainService
from services.chains.openapi_chain import OpenAPIChain

from .openapi_chain import OpenAPIChain


class ChainService:
    def __init__(self, settings: IAgentConfig):
        self._settings = settings
        self._isSQLChainEnabled = False
        self._isOpenAPIChainEnabled = False

    def checkEnabledChains(self, settings: IAgentConfig) -> List[IChain]:
        enabledChains = []

        # if settings.get('dataSourceConfig'):
        #     self._isSQLChainEnabled = True
        #     enabledChains.append(SqlChain(settings['dataSourceConfig']))

        if settings.get('open_api_config'):
            self._isOpenAPIChainEnabled = True
            open_api_chain = OpenAPIChain(settings['open_api_config'])
            enabledChains.append(open_api_chain)

        return enabledChains

    def buildSystemMessages(self, system_messages: str) -> str:
        builtMessage = system_messages

        builtMessage += '\n'
        builtMessage += """
            Given the user prompt and conversation log, the document context, the API output, and the following database output, formulate a response from a knowledge base.\n
            You must follow the following rules and priorities when generating and responding:\n
            - Always prioritize user prompt over conversation record.\n
            - Ignore any conversation logs that are not directly related to the user prompt.\n
            - Only try to answer if a question is asked.\n
            - The question must be a single sentence.\n
            - You must remove any punctuation from the question.\n
            - You must remove any words that are not relevant to the question.\n
            - If you are unable to formulate a question, respond in a friendly manner so the user can rephrase the question.\n\n

            USER PROMPT: {user_prompt}\n
            --------------------------------------
            CHAT HISTORY: {format_chat_messages}\n
            --------------------------------------
            Context found in documents: {summaries}\n
            --------------------------------------
            Name of reference files: {referencies}\n
        """

        if self._isSQLChainEnabled:
            builtMessage += """
                --------------------------------------
                Database Result: {sqlResult}\n
                Query executed: {sqlQuery}\n
                --------------------------------------
            """

        if self._isOpenAPIChainEnabled:
            builtMessage += """
                --------------------------------------
                API Result: {openAPIResult}\n
                --------------------------------------
            """

        return builtMessage

    def buildPromptTemplate(self, system_messages: str) -> BasePromptTemplate:
        combine_messages = [
            SystemMessagePromptTemplate.from_template(self.buildSystemMessages(system_messages)),
            MessagesPlaceholder('chat_history'),
            HumanMessagePromptTemplate.from_template('{question}'),
        ]

        CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(combine_messages)

        return CHAT_COMBINE_PROMPT

    def buildChains(self, llm: BaseChatModel, *args) -> List[Chain]:
        enabledChains = self.checkEnabledChains(self._settings)
        # chainQA = loadQAMapReduceChain(llm, combinePrompt=self.buildPromptTemplate(self._settings.get('system_messsage', SYSTEM_MESSAGE_DEFAULT)))

        # chains = await asyncio.gather(*(chain.create(llm, *args) for chain in enabledChains))
        chains = []
        for chain in enabledChains:
            cn = chain.create(llm, *args)
            chains.append(cn)

        # return chains + [chainQA]
        return chains

    def build(self, llm: BaseChatModel, *args) -> Chain:
        memory = args[0]
        chains = self.buildChains(llm, *args)
        # prompt_template = "Tell me a {question} joke"
        # prompt = PromptTemplate(
        #     input_variables=["question"], template=prompt_template
        # )
        # llm = LLMChain(llm=llm, prompt=prompt)
        # chains = [llm]
        print("🐍 File: chains/chain.py | Line: 109 | build ~ chains",chains)

        enhancementChain = SequentialChain(
            chains=chains,
            input_variables=[
                'query',
                # 'referencies',
                # 'input_documents',
                'question',
                'chat_history',
                'format_chat_messages',
                # 'user_prompt',
            ],
            return_all=True,
            # output_variables=['llm'],
            # memory=memory
        )

        return enhancementChain
