from typing import Optional

from interface.chat_history import IChatHistoryService
from interface.settings import DATABASE_TYPE, IChatHistory
from services.chat_history.memory_chat_history import MemoryChatHistory
from services.chat_history.memory_chat_redis import MemoryChatRedis


class MemoryFactory:
    @staticmethod
    def build(config: IChatHistory, session_id: Optional[str] = None) -> IChatHistoryService:
        memory_type = config.get('type') if config else None
        if memory_type == DATABASE_TYPE.redis:
            return MemoryChatRedis(config, session_id)
        else:
            return MemoryChatHistory()
