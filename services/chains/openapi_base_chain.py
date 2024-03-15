# from langchain.chains import BaseChain, ChainValues
# from interface.agent_interface import IChainInputs
from typing import Dict, Union

from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import (BasePromptTemplate, ChatPromptTemplate,
                               HumanMessagePromptTemplate, MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from openapi3 import OpenAPI

from utils.fetch_helper import fetch


class OpenApiBaseChainInput():
    spec: Union[str, OpenAPI]
    llm: BaseChatModel
    customizeSystemMessage: str
    headers: Dict[str, str]

class OpenApiBaseChain():
    input_keys = 'query'
    output_keys = 'openAPIResult'

    def __init__(self, input) -> None:
        self.input: OpenApiBaseChainInput = input

    @property
    def input_keys(self):
        return [self.input_keys]

    @property
    def output_keys(self):
        return [self.output_keys]

    def getOpenApiPrompt(self):
        return """
        You are an AI with expertise in OpenAPI and Swagger.\n
        You should follow the following rules when generating and answer:\n
        - Only execute the request on the service if the question is not in CHAT HISTORY, if the question has already been answered, use the same answer and do not make a request on the service.
        - Only attempt to answer if a question was posed.\n
        - Always answer the question in the language in which the question was asked.\n
        - The response must be a json object contains an url, contentType, requestMethod and data.\n\n
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
            SystemMessagePromptTemplate(system_messages),
            MessagesPlaceholder('chat_history'),
            HumanMessagePromptTemplate('{question}'),
        ]

        return ChatPromptTemplate(combine_messages)

    def _call(self, values):
        print("🐍 File: chains/openapi_base_chain.py | Line: 66 | buildPromptTemplate ~ values",values)
        question = values[self.input_keys]
        schema = self.input.spec

        fetch_sentence = {
            "schema": lambda: schema,
            "question": lambda input: input["question"],
            "chat_history": lambda: values.get("chat_history"),
            "format_chat_messages": lambda: values.get("format_chat_messages"),
            # "user_prompt": lambda: self.input.customizeSystemMessage
            } | self.buildPromptTemplate(self.getOpenApiPrompt()) | self.input.llm.bind({})

        final_chain = {
            "question": lambda input: input["question"], "query": fetch_sentence
            } | {
            "table_info": lambda: schema, "input": lambda: question, "schema": lambda: schema, "question": lambda input: input["question"], "query": lambda input: input["query"], "response": lambda input: fetch(input["query"]["content"], self.input.headers)
            } | { self.output_keys: lambda input: input["response"]["body"] } | StrOutputParser()

        result = final_chain.invoke({"question": question})
        return result

    def _chainType(self) -> str:
        return 'open_api_chain'
