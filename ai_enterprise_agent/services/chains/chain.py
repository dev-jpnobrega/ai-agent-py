from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel

from ai_enterprise_agent.interface.settings import CHAIN_TYPE, ISettings
from ai_enterprise_agent.services.chains.open_api_chain import OpenApiChain
from ai_enterprise_agent.services.chains.simple_chain import SimpleChain
from ai_enterprise_agent.services.chains.sql_chain import SqlChain
from ai_enterprise_agent.services.chains.vector_store_chain import \
    VectorStoreChain


class ChainFactory:
    """
    Factory class to build different types of chains based on configuration.
    """

    @staticmethod
    def build(chain_type: str, config: ISettings, model: BaseChatModel, memory: ConversationBufferMemory = None) -> Chain:
        """
        Build a chain based on the provided chain type.

        Args:
            chain_type (str): The type of chain to build.
            config (ISettings): Configuration for the chain.
            model (BaseChatModel): The chat model to use in the chain.
            memory (ConversationBufferMemory): Memory buffer for the chain.

        Returns:
            Chain: An instance of the specified chain type.

        Raises:
            ValueError: If an invalid chain type is provided.
        """
        chain_classes = {
            CHAIN_TYPE.open_api_chain: OpenApiChain,
            CHAIN_TYPE.simple_chain: SimpleChain,
            CHAIN_TYPE.sql_chain: SqlChain,
            CHAIN_TYPE.vector_store_chain: VectorStoreChain
        }

        chain_class = chain_classes.get(chain_type)
        if chain_class:
            return chain_class(config=config, model=model, memory=memory)
        else:
            raise ValueError(f'Invalid chain type: {chain_type}')
