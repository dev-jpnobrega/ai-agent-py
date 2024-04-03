import asyncio
from asyncio import Task
from typing import Any, Dict, List

from langchain.chains.base import Chain

from ai_enterprise_agent.interface.settings import ISettings


class CustomSequentialChain(Chain):

  input_key = 'question'
  output_key = 'sequential_chain'
  config: Dict[str, Any] = None
  chains: List[Chain] = None

  @property
  def input_keys(self):
    return [self.input_key]

  @property
  def output_keys(self):
    return [self.output_key]

  def __init__(self, config: ISettings, chains: List[Dict[str, Chain]]):
    super().__init__()
    self.config = config
    self.chains = chains

  def chain(self, input: Dict[str, Any]) -> List[Task]:
    tasks: List[Task] = []
    for chain in self.chains:
      task = asyncio.create_task(chain['chain']._call(input=input))
      tasks.append(task)
    return tasks

  async def _call(self, input: Dict[str, Any]):
    chain = self.chain(input)
    response = await asyncio.gather(*chain)
    return response

  @property
  def _chain_type(self) -> str:
    return "sequential_chain"
