from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai_enterprise_agent.interface.settings import (CHAIN_TYPE,
                                                    PROCESSING_TYPE, ISettings)
from ai_enterprise_agent.services.chains.chain import ChainFactory
from ai_enterprise_agent.services.chains.orchestrator_chain import \
    OrchestratorChain
from ai_enterprise_agent.services.chains.sequential_chain import \
    CustomSequentialChain
from ai_enterprise_agent.services.chat_history.memory import MemoryFactory
from ai_enterprise_agent.services.llm.model import ModelFactory


class Agent:

  def __init__(self, config: ISettings) -> None:
    self.config = config
    self.model = ModelFactory.build(config.get('model'))
    self.memory = None

  def validate_configuration(self):
    if self.config.get('processing_type') == None or self.config.get('chains') == None:
      raise ValueError('Processing type and chains must be specified')

    chains = self.config.get('chains')
    chains_length = len(chains)
    if (self.config.get('processing_type') == PROCESSING_TYPE.single and chains_length > 1) or (self.config.get('processing_type') in [PROCESSING_TYPE.sequential, PROCESSING_TYPE.orchestrated] and chains_length == 1):
      raise ValueError(f"Invalid processing type: {self.config.get('processing_type').value} and number of chains: {chains_length}")

  def build_chains(self):
    self.validate_configuration()
    chains: List[CHAIN_TYPE] = self.config.get('chains')

    if self.config.get('processing_type') == PROCESSING_TYPE.single:
      return ChainFactory.build(chains[0], self.config, model=self.model, memory=self.memory)

    enabled_chains = [{'name': chain, 'chain': ChainFactory.build(chain, self.config, model=self.model, memory=self.memory)} for chain in chains]

    if self.config.get('processing_type') == PROCESSING_TYPE.sequential:
      return CustomSequentialChain(config=self.config, chains=enabled_chains)

    if self.config.get('processing_type') == PROCESSING_TYPE.orchestrated:
      return OrchestratorChain(config=self.config, model=self.model, chains=enabled_chains)

  def build_system_messages(self):
    config = self.config
    message = """
      Given the user prompt and conversation log, the document context, the API output, and the following database output, formulate a response from a knowledge base.\n
      You must follow the following rules and priorities when generating and responding:\n
      - Always prioritize user prompt over conversation record.\n
      - Ignore any conversation logs that are not directly related to the user prompt.\n
      - Only try to answer if a question is asked.\n
      - The question must be a single sentence.\n
      - You must remove any punctuation from the question.\n
      - You must remove any words that are not relevant to the question.\n
      - If you are unable to formulate a question, respond in a friendly manner so the user can rephrase the question.\n\n
      Question: {question}
    """

    if CHAIN_TYPE.simple_chain in config.get('chains'):
      message += """
        LLM Result: {simple_chain}\n
      """

    if CHAIN_TYPE.sql_chain in config.get('chains'):
      message += """
        SQL Result: {sql_chain}\n
      """

    if CHAIN_TYPE.open_api_chain in config.get('chains'):
      message += """
        OPEN API Result: {open_api_chain}\n
      """

    if CHAIN_TYPE.vector_store_chain in config.get('chains'):
      message += """
        Document Result: {vector_store_chain}\n
      """

    return message

  def refine_result(self, input: Dict[str, Any], result: Any | List):
    map_prompt = PromptTemplate.from_template(self.build_system_messages())
    map_chain = map_prompt | self.model | StrOutputParser()
    response = map_chain.invoke(input={
      "question": input.get('question'),
      "simple_chain": result.get('simple_chain', None),
      "sql_chain": result.get('sql_chain', None),
      "open_api_chain": result.get('open_api_chain', None),
      "vector_store_chain": result.get('vector_store_chain', None),
    })
    return response

  async def _call(self, input: Dict[str, Any]):
    config = self.config
    self.memory = MemoryFactory.build(config.get('history'), input.get('chat_thread_id'))
    chain = self.build_chains()
    result = await chain._call(input)
    if config.get('processing_type') == PROCESSING_TYPE.sequential:
      result = self.refine_result(input, result[0])

    self.memory.add_user_message(message=input.get('question'))
    self.memory.add_ai_message(message=result)

    return result
