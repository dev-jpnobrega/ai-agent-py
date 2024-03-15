from langchain.chat_models.base import BaseChatModel

from interface.agent_interface import LLM_TYPE, IChatConfig, ILLMConfig
from services.llm.azure_llm import AzureLLMService
from services.llm.bedrock_llm import BedrockLLMService
from services.llm.google_llm import GoogleLLMService


class LLMFactory:
    """
    Factory class for creating language model instances.
    """

    @staticmethod
    def create(chat_settings: IChatConfig, llm_settings: ILLMConfig) -> BaseChatModel:
        """
        Create a language model instance based on the given settings.

        Args:
            chat_settings (IChatConfig): Chat configuration settings.
            llm_settings (ILLMConfig): Language model configuration settings.

        Returns:
            BaseChatModel: Instance of the created language model.
        """
        llm_type = llm_settings['type']

        if llm_type == LLM_TYPE.aws:
            llm_service = AzureLLMService(chat_settings, llm_settings)
        elif llm_type == LLM_TYPE.google:
            llm_service = GoogleLLMService(chat_settings, llm_settings)
        elif llm_type == LLM_TYPE.azure:
            llm_service = BedrockLLMService(chat_settings, llm_settings)
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")

        try:
            return llm_service.build()
        except Exception as e:
            raise RuntimeError("Failed to create language model") from e
