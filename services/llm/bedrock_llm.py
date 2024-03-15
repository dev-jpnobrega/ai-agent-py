from langchain.chat_models.base import BaseChatModel
from interface.agent_interface import IChatConfig, ILLMConfig
from langchain.chat_models.bedrock import BedrockChat

class BedrockLLMService:
    def __init__(self, chat_settings: IChatConfig, llm_settings: ILLMConfig):
        self._chat_settings = chat_settings
        self._llm_settings = llm_settings

    def build(self) -> BaseChatModel:
        return BedrockChat(
            temperature=self._chat_settings['temperature'],
            streaming=True,
            model=self._llm_settings['model'],
            region=self._llm_settings['region'],
            credentials={
                'accessKeyId': self._llm_settings['api_key'],
                'secretAccessKey': self._llm_settings['secret_access_key'],
                'sessionToken': self._llm_settings['session_token']
            }
        )
