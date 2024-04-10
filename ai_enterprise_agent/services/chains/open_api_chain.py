import json
from typing import Any, Dict

from langchain.chains.base import Chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable

from ai_enterprise_agent.interface.chat_history import IChatHistoryService
from ai_enterprise_agent.interface.settings import PROCESSING_TYPE, ISettings
from ai_enterprise_agent.utils.fetch_helper import fetch


class OpenApiChain(Chain):

  input_key = 'question'
  output_key = 'open_api_chain'
  model: BaseChatModel = None
  memory: IChatHistoryService = None
  open_api: Dict[str, Any] = None
  config: ISettings = None

  @property
  def input_keys(self):
    return [self.input_key]

  @property
  def output_keys(self):
    return [self.output_key]

  def __init__(self, config: ISettings,  model: BaseChatModel, memory: IChatHistoryService) -> None:
    super().__init__()
    self.model = model
    self.memory = memory
    self.open_api = config.get('open_api')
    self.config = config

  def get_fetch(self, question, custom_system_message) -> str:
    open_api = self.open_api
    schema = open_api.get('data')
    template = """
      You are an AI with expertise in OpenAPI and Swagger.
      You should follow the following rules when generating an answer:
      - Only execute the request on the service if the question is not in History, if the question has already been answered, use the same answer and do not make a request on the service.
      - The response must be a JSON object containing an url, content type, method, and data, without triple quotes, json string on start and the end.
      {custom_system_message}
      -------------------------------------------\n
      Schema: {schema}\n
      -------------------------------------------\n
      History: {history}\n
      -------------------------------------------\n
      Question: {question}\n
      ------------------------------------------\n
      API ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | self.model | StrOutputParser()
    return chain.invoke({"schema": schema, "question": question, "history": self.memory.get_messages(), "custom_system_message": custom_system_message})

  def chain(self, question, custom_system_message) -> RunnableSerializable[Any, Any]:
    fetch_sentence = self.get_fetch(question, custom_system_message)
    request = json.loads(fetch_sentence)
    response = fetch(url=request.get('url'), method=request.get('method'), data=request.get('data'), headers={})
    template = """
        Based on the context below, answer the question with natural language.
        You should follow the following rules when generating and answer:
        - Only attempt to answer if a question was posed.
        - Always answer the question in the language in which the question was asked.
        Context: {context}
        Question: {question}
      """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (RunnablePassthrough.assign(context=lambda _: response)) | prompt | self.model | StrOutputParser()
    return chain

  async def _call(self, input: Dict[str, Any] = None):
    try:
      question = input['question']
      custom_system_message = input.get('custom_system_message', None)
      chain = self.chain(question, custom_system_message)
      result = chain.invoke(input={'question': question})
      if self.config.get('processing_type') == PROCESSING_TYPE.sequential:
        return { self.output_key: result }
      return result
    except ValueError as e:
      result = str(e)
      if not result.startswith("Could not parse LLM output: `"):
          raise e
      result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")

  @property
  def _chain_type(self) -> str:
    return "open_api_chain"
