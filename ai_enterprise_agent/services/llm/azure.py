from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from ai_enterprise_agent.interface.settings import IModel


class AzureChatModel:
    """
    Represents an Azure Chat Model.
    """

    def __init__(self, model_config: IModel) -> None:
        """
        Initialize the Azure Chat Model with the provided configuration.

        Args:
            model_config (IModel): The configuration for the model.
        """
        self.temperature = model_config.get('temperature', 0.5)
        self.azure_deployment = model_config.get('model')
        self.api_key = model_config.get('api_key')
        self.api_version = model_config.get('api_version')
        self.azure_endpoint = model_config.get('endpoint')
        self.streaming = True

    def build(self) -> BaseChatModel:
        """
        Build and return an Azure Chat OpenAI model.

        Returns:
            BaseChatModel: An instance of the Azure Chat OpenAI model.
        """
        azure_endpoint = f'https://{self.azure_endpoint}.openai.azure.com/'
        return AzureChatOpenAI(
            temperature=self.temperature,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            api_key=self.api_key,
            azure_endpoint=azure_endpoint,
            streaming=self.streaming
        )
