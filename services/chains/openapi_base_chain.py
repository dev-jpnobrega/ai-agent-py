import json
from typing import Any, Dict, List, Optional, Union

import langchain.memory
from langchain.callbacks.manager import (AsyncCallbackManagerForChainRun,
                                         CallbackManagerForChainRun)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (BasePromptTemplate, ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    MessagesPlaceholder, PromptTemplate,
                                    SystemMessagePromptTemplate)
from openapi3 import OpenAPI
from pydantic import Extra

from utils.fetch_helper import fetch


class OpenApiBaseChainInput():
    spec: Union[str, OpenAPI]
    llm: BaseChatModel
    customizeSystemMessage: str
    headers: Dict[str, str]
    memory: ConversationBufferMemory

class OpenApiBaseChain(Chain):
    # input: Dict[str, Any]
    # output_keys = 'openAPIResult'
    # memory = None

    # @property
    # def input(self):
    #     return self.input

    # class Config:
    #     """Configuration for this pydantic object."""

    #     extra = Extra.forbid
    #     arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.question_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def __fields_set__(self):
        pass

    def __init__(self) -> None:
        super().__init__()
        # self.input_keys = ['query', 'question', 'chat_history', 'format_chat_messages']
        # self.output_keys = ['openAPIResult']
        # self.memory = None
        # self.input = input
        # print("🐍 File: chains/openapi_base_chain.py | Line: 43 | __init__ ~ input",input)

    def getOpenApiPrompt(self):
        return """
        You are an AI with expertise in OpenAPI and Swagger.\n
        You should follow the following rules when generating and answer:\n
        - Only execute the request on the service if the question is not in CHAT HISTORY, if the question has already been answered, use the same answer and do not make a request on the service.
        - Only attempt to answer if a question was posed.\n
        - Always answer the question in the language in which the question was asked.\n
        - The response must be a json object contains an url, content_type, method and data.\n\n
        -------------------------------------------\n
        SCHEMA: {schema}\n
        -------------------------------------------\n
        CHAT HISTORY: {format_chat_messages}\n
        -------------------------------------------\n
        QUESTION: {question}\n
        ------------------------------------------\n
        API ANSWER:
        """

    def buildPromptTemplate(self, system_messages: str) -> BasePromptTemplate:
        combine_messages = [
            SystemMessagePromptTemplate.from_template(system_messages),
            MessagesPlaceholder('chat_history'),
            HumanMessagePromptTemplate.from_template('{question}'),
        ]

        return ChatPromptTemplate.format_messages(combine_messages)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None):
        prompt = PromptTemplate(input_variables=["schema", "question", "format_chat_messages"], template=self.getOpenApiPrompt())
        chain = LLMChain(llm=inputs.get('llm'), prompt=prompt, output_key='request')
        result = chain.invoke(
            input={
                "schema": inputs.get('spec'),
                "question": inputs.get('question'),
                "format_chat_messages": inputs.get("format_chat_messages"),
            }
        )
        request = json.loads(result.get("request"))
        headers = self.input.get('headers') if self.input.get('headers') else {}
        custom_headers = {
            **headers,
            'Content-Type': request.get('content_type'),
        }
        response = fetch(
            url=request.get('url'),
            method=request.get('method'),
            data=request.get('data'),
            headers=custom_headers,
        )
        return { 'openAPIResult': response }

    # async def _acall(
    #         self,
    #         inputs: Dict[str, Any],
    #         run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    # ) -> Dict[str, str]:
    #     raise NotImplementedError("Does not support async")
    @property
    def _chain_type(self) -> str:
        return 'open_api_chain'
