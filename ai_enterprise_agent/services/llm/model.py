from langchain_core.language_models.chat_models import BaseChatModel

from ai_enterprise_agent.interface.settings import LLM_TYPE, IModel
from ai_enterprise_agent.services.llm.azure import AzureChatModel
from ai_enterprise_agent.services.llm.google import GoogleChatModel


class ModelFactory:
    @staticmethod
    def build(config: IModel) -> BaseChatModel:
        """
        Build a chat model based on the provided configuration.

        Args:
            config (IModel): The configuration for the model.

        Returns:
            BaseChatModel: An instance of the chat model.

        Raises:
            ValueError: If the model type is invalid.
        """
        model_type = config.get('type')
        if model_type == LLM_TYPE.google:
            chat_model = GoogleChatModel(config)
        elif model_type == LLM_TYPE.azure:
            chat_model = AzureChatModel(config)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))
        return chat_model.build()
