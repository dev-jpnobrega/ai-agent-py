from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel

from interface.settings import CHAIN_TYPE, ISettings
from services.chains.open_api_chain import OpenApiChain
from services.chains.simple_chain import SimpleChain
from services.chains.sql_chain import SqlChain


class ChainFactory:
    """
    Factory class to build different types of chains based on configuration.
    """

    @staticmethod
    def build(chain_type: str, config: ISettings, model: BaseChatModel, memory: ConversationBufferMemory) -> Chain:
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
            CHAIN_TYPE.sql_chain: SqlChain
        }

        chain_class = chain_classes.get(chain_type)
        if chain_class:
            return chain_class(config=config, model=model, memory=memory)
        else:
            raise ValueError(f'Invalid chain type: {chain_type}')
