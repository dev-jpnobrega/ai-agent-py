from interface.agent_interface import IDatabaseConfig
from interface.chat_history_interface import IChatHistory
from services.chat_history.memory_chat_history import MemoryChatHistory
from services.chat_history.redis_chat_history import RedisChatHistory


class ChatHistoryFactory:
    MEMORY = 'memory'
    REDIS = 'redis'

    @staticmethod
    def create(settings: IDatabaseConfig) -> IChatHistory:
        service_type = settings.get('type')

        if service_type == ChatHistoryFactory.REDIS:
            return RedisChatHistory(settings)
        elif service_type == ChatHistoryFactory.MEMORY:
            return MemoryChatHistory(settings)
        else:
            # Se o tipo de serviço não for reconhecido, retorne uma instância do histórico de chat na memória como padrão
            print(f"Tipo de serviço desconhecido: {service_type}. Usando MemoryChatHistory como padrão.")
            return MemoryChatHistory(settings)
