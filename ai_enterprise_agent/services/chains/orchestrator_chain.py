from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_enterprise_agent.interface.settings import CHAIN_TYPE, ISettings


class OrchestratorChain(Chain):

  input_key = 'question'
  output_key = 'orchestrator_chain'
  model: BaseChatModel = None
  config: Dict[str, Any] = None
  chains: List[Chain] = None
  open_api_schema: Dict[str, Any] = None
  database_schema: Dict[str, Any] = None
  documents: Dict[str, Any] = None

  @property
  def input_keys(self):
    return [self.input_key]

  @property
  def output_keys(self):
    return [self.output_key]

  def __init__(self, config: ISettings, model: BaseChatModel, chains: List[Dict[str, Chain]]):
    super().__init__()
    self.config = config
    self.model = model
    self.chains = chains

  def build_knowledge(self, question: Optional[str]):
    has_sql_chain = next((item for item in self.chains if item["name"] == CHAIN_TYPE.sql_chain), None)
    has_open_api_chain = next((item for item in self.chains if item["name"] == CHAIN_TYPE.open_api_chain), None)
    has_vector_store_chain = next((item for item in self.chains if item["name"] == CHAIN_TYPE.vector_store_chain), None)
    if has_sql_chain:
      self.database_schema = has_sql_chain['chain'].get_schema(None)
    if has_open_api_chain:
      schema = self.config.get('open_api')
      self.open_api_schema = schema.get('data')
    if has_vector_store_chain:
      self.documents = has_vector_store_chain['chain'].build_relevant_docs(question)

  def chain(self, input: Dict[str, Any]) -> Chain:
    question = input['question']
    self.build_knowledge(question)
    prompt_chain = (
      PromptTemplate.from_template(
        """Given the user question below, identify what's the better chain we can use to answer the question.
          To help identify, consider the following features about our chains:
            - open_api_chain - You are an AI with expertise in OpenAPI and Swagger.
            - Based on the OpenAPI or Swagger schema below, this chain can be used to answer the question about these schema subjects.
            - {schema}
            - sql_chain - You are an AI with expertise in create SQL Sentences.
            - Based on the table schema below, this chain can be used to answer the question about these schema subjects.
            - {database_schema}
            - vector_store_chain - You are an AI with expertise in document analysis.
            - Based on the documents below, this chain can be used to answer the question about these document subjects.
            - {documents}
            - simple_chain - You are an AI with general knowledge.
          Do not respond with more than one word.\n
          <question>
          {question}
          Better choice:"""
        )
        | self.model
        | StrOutputParser()
      )
    response = prompt_chain.invoke({"question": question, "database_schema": self.database_schema, "schema": self.open_api_schema, "documents": self.documents})
    final_chain = next(item for item in self.chains if item["name"] == CHAIN_TYPE[response.strip()])
    return final_chain['chain']

  async def _call(self, input: Dict[str, Any]):
    chain = self.chain(input)
    response = await chain._call(input)
    return response

  @property
  def _chain_type(self) -> str:
    return "orchestrator_chain"
