from typing import Any, Dict

from langchain.chains.base import Chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ai_enterprise_agent.interface.chat_history import IChatHistoryService
from ai_enterprise_agent.interface.settings import PROCESSING_TYPE, ISettings
from ai_enterprise_agent.services.vector_store.vector_store import \
    VectorStoreFactory


class VectorStoreChain(Chain):

  input_key = 'question'
  output_key = 'vector_store_chain'
  memory: IChatHistoryService = None
  model: BaseChatModel = None
  config: ISettings = None
  vector_store: VectorStore = None

  @property
  def input_keys(self):
    return [self.input_key]

  @property
  def output_keys(self):
    return [self.output_key]

  def __init__(self, config: ISettings, model: BaseChatModel, memory: IChatHistoryService):
    super().__init__()
    self.model = model
    self.memory = memory
    self.config = config
    self.vector_store = VectorStoreFactory.build(config=config, model=self.model)

  def build_relevant_docs(self, question: str, k: int = 10):
    return self.vector_store.similarity_search(query=question, k=k)

  def chain(self):
    pass

  async def _call(self, input: Dict[str, Any]):
    question = input.get('question')
    response = self.vector_store._call(question, False)
    if self.config.get('processing_type') == PROCESSING_TYPE.sequential:
      return { self.output_key: response.get('result') }
    return response.get('result')
