from typing import Any, Dict


class IAgent:
  def chain(self, input: Dict[str, Any]):
    ...
  def _call(self):
    ...
