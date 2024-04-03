from typing import Any, Dict, Optional, Sequence

from langchain_core.messages import BaseMessage


class IChatHistoryService:
  def get_memory(self):
    pass

  def add_user_message(self, message: str):
    pass

  def add_ai_message(self, message: str):
    pass

  def add_message(self, message: BaseMessage) -> None:
    pass

  def add_messages(self, messages: Sequence[BaseMessage]) -> None:
    pass

  def load_memory_variables(self, inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pass

  def get_messages(self):
    pass

  def clear(self):
    pass
