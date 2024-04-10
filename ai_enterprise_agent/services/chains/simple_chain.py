from operator import itemgetter
from typing import Any, Dict

from langchain.chains.base import Chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable

from ai_enterprise_agent.interface.chat_history import IChatHistoryService
from ai_enterprise_agent.interface.settings import PROCESSING_TYPE, ISettings


class SimpleChain(Chain):

  input_key = 'question'
  output_key = 'simple_chain'
  model: BaseChatModel = None
  memory: IChatHistoryService = None
  config: ISettings = None

  @property
  def input_keys(self):
    return [self.input_key]

  @property
  def output_keys(self):
    return [self.output_key]

  def __init__(self, config: ISettings, model: BaseChatModel, memory: IChatHistoryService) -> None:
    super().__init__()
    self.model = model
    self.memory = memory
    self.config = config

  def build_system_messages(self, customize_system_message: str) -> str:
    message = """
      You are an AI with general knowledge.\n
      Only execute the request on the service if the question is not in History, if the question has already been answered, use the same answer and not request the service.\n
      History: {history}
      Question: {question}
    """
    if customize_system_message:
      message += customize_system_message

    return message

  def chain(self) -> RunnableSerializable[Any, Any]:
    settings = self.config.get('system')
    prompt = ChatPromptTemplate.from_template(self.build_system_messages(settings.get('system_message')))
    return  RunnablePassthrough.assign(history=RunnableLambda(self.memory.get_messages()) | itemgetter("history")) | prompt | self.model | StrOutputParser()

  async def _call(self, input: Dict[str, Any]):
    chain = self.chain()
    response = chain.invoke(input)
    if self.config.get('processing_type') == PROCESSING_TYPE.sequential:
      return  { self.output_key: response }
    return response

  @property
  def _chain_type(self) -> str:
    return "simple_chain"
