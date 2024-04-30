from typing import Any, Dict

from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

  def build_relevant_docs(self, query: str, k: int = 10):
    config = self.config.get('vector_store')
    return self.vector_store.similarity_search(query=query, k=k, filters=config.get('custom_filters', None))

  def chain(self, query: str):
    context = self.build_relevant_docs(query)
    template = """Use the following the context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    return (
      RunnablePassthrough.assign(context=lambda _: context)
      | prompt
      | self.model
      | StrOutputParser()
    )

  async def _call(self, input: Dict[str, Any]):
    question = input.get('question')
    chain = self.chain(question)
    response = chain.invoke(input={"question": question})
    if self.config.get('processing_type') == PROCESSING_TYPE.sequential:
      return { self.output_key: response }
    return response
