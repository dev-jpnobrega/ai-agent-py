from langchain_google_genai import ChatGoogleGenerativeAI
from interface.agent_interface import IChatConfig, ILLMConfig

class GoogleLLMService:
    """
    Service class for building a Google Generative AI chat model.
    """

    def __init__(self, chat_settings: IChatConfig, llm_settings: ILLMConfig):
        """
        Initialize the GoogleLLMService with chat and LLM settings.

        Args:
            chat_settings (IChatConfig): Chat configuration settings.
            llm_settings (ILLMConfig): LLM configuration settings.
        """
        self._chat_settings = chat_settings
        self._llm_settings = llm_settings

    def build(self) -> ChatGoogleGenerativeAI:
        """
        Build and return the Google Generative AI chat model.

        Returns:
            ChatGoogleGenerativeAI: Instance of the built chat model.
        """
        temperature = self._chat_settings['temperature']
        model = self._llm_settings['model']
        google_api_key = self._llm_settings['api_key']
        max_tokens = self._chat_settings['max_tokens'] or 2048
        streaming = True

        try:
            # Validating input
            if not temperature or not model or not google_api_key:
                raise ValueError("Invalid settings: temperature, model, and api_key are required")

            return ChatGoogleGenerativeAI(
                temperature=temperature,
                model=model,
                google_api_key=google_api_key,
                max_output_tokens=max_tokens,
                streaming=streaming
            )
        except Exception as e:
            raise RuntimeError("Failed to build Google Generative AI chat model") from e
