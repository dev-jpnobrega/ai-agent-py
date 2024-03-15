import asyncio
import json
from collections import namedtuple
from typing import Any, Dict, NamedTuple, Union

from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from nanoid import generate

from interface.agent_interface import IOpenAPIConfig
from utils.fetch_helper import fetch

from .chain import IChain


class OpenApiBaseChainInput():
    spec: Union[str]
    llm: BaseChatModel
    customizeSystemMessage: str
    headers: Dict[str, str]
    timeout: int

class OpenAPIBaseChain():
    inputKey = 'query'
    outputKey = 'openAPIResult'

    def __init__(self, llm: BaseChatModel, spec: str, customize_system_message: str = None, headers: Dict[str, Any] = None, timeout: int = None) -> None:
        # super().__init__(input)
        # self._input: OpenApiBaseChainInput = input
        self._llm = llm
        self._spec = spec
        self._customize_system_message = customize_system_message
        self._headers = headers
        self._timeout = timeout

    def _call(self):
        fetch_prompt = PromptTemplate.from_template(
            '''You are an AI with expertise in OpenAPI and Swagger.\n
            You should follow the following rules when generating and answer:\n
            - Only attempt to answer if a question was posed.\n
            - Always answer the question in the language in which the question was asked.\n
            - The response must be a json object contains an url, content_type, request_method and data.\n
            {customize_system_message}
            SCHEMA: {schema}\n
            QUESTION: {question}\n
            API ANSWER:
            '''
        )
        query = "Crie um chat com o agentUid 2764c812-3fe2-4982-88ad-c1df679ead7c, userId 2764c812-3fe2-4982-88ad-c1df679ead7c, mandando via POST a seguinte mensagem - Forneca informacoes sobre a aplicacao de imposto no GSP?"
        answer = fetch_prompt | self._llm | StrOutputParser()
        request = answer.invoke({"question": query, "schema": self._spec, "customize_system_message": self._customize_system_message})
        formatted_request = json.loads(request)

        # answer_prompt = '''
        #     You are an AI with expertise in OpenAPI and Swagger.\n
        #     You should follow the following rules when generating and answer:\n
        #     - Only execute the request on the service if the question is not in CHAT HISTORY, if the question has already been answered, use the same answer and do not make a request on the service.
        #     - Only attempt to answer if a question was posed.\n
        #     - Always answer the question in the language in which the question was asked.\n
        #     - The response must be a json object contains an url, contentType, requestMethod and data.\n\n
        #     -------------------------------------------\n
        #     SCHEMA: {schema}\n
        #     -------------------------------------------\n
        #     QUESTION: {question}\n
        #     ------------------------------------------\n
        #     API ANSWER:
        #     '''

        # Params = NamedTuple('Params', 'Parameters')
        # param_mapping = {
        #     Params('query_params',['query_params']),
        #     Params('body_params',['body_params']),
        #     Params('path_params',['path_params']),
        #     # NamedTuple("body_params", [('name', 'id')]),
        #     # NamedTuple("path_params", [('name', 'id')]),
        # }
        # param_mapping = _ParamMapping(
        #     query_params=operation.query_params,
        #     body_params=operation.body_params,
        #     path_params=operation.path_params,
        # )

        # response = OpenAPIEndpointChain.from_api_operation(
        #     operation={
        #         'base_url': formatted_request.get('url'),
        #         'path': '',
        #         'method': str(formatted_request.get('request_method')).lower(),
        #         'request_body': {
        #             'content_type': formatted_request.get('content_type'),
        #             'data': formatted_request.get('data'),
        #             'media_type': formatted_request.get('media_type', ''),
        #             "properties": [],
        #         },
        #         "operation_id": generate(),
        #         "properties": [],
        #     },
        #     llm=self._llm
            # api_operation={
            #     'base_url': formatted_request.get('url'),
            #     'path': '',
            #     'method': str(formatted_request.get('request_method')).lower(),
            #     'request_body': {
            #         'content_type': formatted_request.get('content_type'),
            #         'data': formatted_request.get('data'),
            #         'media_type': formatted_request.get('media_type', ''),
            #         "properties": [],
            #     },
            #     "operation_id": generate(),
            #     "properties": [],
            # },
            # api_request_chain={
            #     "llm": self._llm,
            #     "prompt": PromptTemplate(input=query, input_variables=['question', 'schema'], template=answer_prompt, template_format='f-string'),
            # },
            # param_mapping=param_mapping,
            # verbose=True
        # )
        # response = OpenAPIEndpointChain.from_api_operation(
        #     llm=self._llm,
        #     operation={
        #         "operation_id": generate(),
        #         "base_url": formatted_request.get('url'),
        #         "method": str(formatted_request.get('request_method')).lower(),
        #         "request_body": formatted_request.get('data'),
        #         "path": "",
        #     },

        # )
        # return response.invoke({
        #     'question': query,
        #     'schema': self._spec,
        # })
        answer = fetch_prompt | self._llm | StrOutputParser()
        request = answer.invoke({"question": query, "schema": self._spec, "customize_system_message": self._customize_system_message})
        formatted_request = json.loads(request)

        custom_headers = {
            # **self._headers,
            # **formatted_request.get('content_type')
        }

        response = asyncio.run(
            fetch(
                url=formatted_request.get('url'),
                method=formatted_request.get('request_method'),
                headers=custom_headers,
                data=formatted_request.get('data')
            )
        )
        return response

    @property
    def inputKeys(self):
        return [self.inputKey]

    @property
    def outputKeys(self):
        return [self.outputKey]

    def _chainType(self) -> str:
        return 'open_api_chain'
