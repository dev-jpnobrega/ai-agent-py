from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from ai_enterprise_agent.interface.settings import IModel


class GoogleChatModel:
    """
    Represents a Google Chat Model.
    """

    def __init__(self, model_config: IModel) -> None:
        """
        Initialize the Google Chat Model with the provided configuration.

        Args:
            model_config (IModel): The configuration for the model.
        """
        self.temperature = model_config.get('temperature', 0.5)
        self.model = model_config.get('model', 'gemini-pro')
        self.google_api_key = model_config.get('api_key')
        self.streaming = True
        self.convert_system_message_to_human = True

    def build(self) -> BaseChatModel:
        """
        Build and return a Google Chat Generative AI model.

        Returns:
            BaseChatModel: An instance of the Google Chat Generative AI model.
        """
        return ChatGoogleGenerativeAI(
            temperature=self.temperature,
            model=self.model,
            google_api_key=self.google_api_key,
            streaming=self.streaming,
            convert_system_message_to_human=self.convert_system_message_to_human
        )
