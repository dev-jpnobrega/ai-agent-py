from typing import Any, Dict, List, Optional, Sequence

from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseChatMessageHistory
from langchain_community.chat_message_histories.redis import \
    RedisChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from nanoid import generate

from ai_enterprise_agent.interface.chat_history import IChatHistoryService
from ai_enterprise_agent.interface.settings import IChatHistory


class MemoryChatRedis(BaseChatMessageHistory, IChatHistoryService):

  memory_key: str = "history"
  chat_history: IChatHistory = None

  @property
  def memory_variables(self) -> List[str]:
    return [self.memory_key]

  def __init__(self, config: IChatHistory, session_id: str):
    super().__init__()
    self.config = config
    self.session_id = session_id
    self.chat_history = self.get_client()
    self.memory = ConversationBufferWindowMemory(
        return_messages=True,
        memory_key='history',
        chat_memory=self.chat_history,
        k=config.get('limit')
    )

  def get_client(self) -> RedisChatMessageHistory:
    if self.chat_history:
      return self.chat_history

    config = self.config
    url = f"redis://:{config.get('password')}@{config.get('host')}:{int(config.get('port'))}/{int(config.get('database',0))}"
    chat_memory = RedisChatMessageHistory(
      session_id=self.session_id or generate(),
      ttl=config.get('session_ttl'),
      key_prefix='history',
      url=url
    )

    return chat_memory

  def get_memory(self):
        return self.memory

  def add_user_message(self, message: str):
      if isinstance(message, HumanMessage):
          self.add_message(message)
      else:
          self.add_message(HumanMessage(content=message))

  def add_ai_message(self, message: str):
      if isinstance(message, AIMessage):
          self.add_message(message)
      else:
          self.add_message(AIMessage(content=message))

  def add_message(self, message: BaseMessage) -> None:
    if type(self).add_messages != BaseChatMessageHistory.add_messages:
        self.chat_history.add_message(message)
    else:
        raise NotImplementedError(
            "add_message is not implemented for this class. "
            "Please implement add_message or add_messages."
        )

  def add_messages(self, messages: Sequence[BaseMessage]) -> None:
    pass

  def load_memory_variables(self, inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return { self.memory_key: self.chat_history.messages }

  def get_messages(self):
    return self.load_memory_variables

  def clear(self):
    pass
