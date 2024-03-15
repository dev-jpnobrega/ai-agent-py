from langchain.chat_models.base import BaseChatModel
from interface.agent_interface import IChatConfig, ILLMConfig
from langchain.chat_models.openai import ChatOpenAI


class AzureLLMService:
    def __init__(self, chat_settings: IChatConfig, llm_settings: ILLMConfig):
        self._chat_settings = chat_settings
        self._llm_settings = llm_settings

    def build(self) -> BaseChatModel:
        return ChatOpenAI(
            temperature=self._chat_settings['temperature'],
            streaming=True,
            azureOpenAIApiDeploymentName=self._llm_settings['model'],
            azureOpenAIApiVersion=self._llm_settings['api_version'],
            azureOpenAIApiKey=self._llm_settings['api_key'],
            azureOpenAIApiInstanceName=self._llm_settings['instance']
        )
