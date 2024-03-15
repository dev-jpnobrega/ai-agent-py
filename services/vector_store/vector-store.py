from langchain.vectorstores.base import VectorStore
from azure_vector_store import AzureCogSearch
from interface.agent_interface import ILLMConfig, IVectorStoreConfig
from langchain.embeddings.openai import OpenAIEmbeddings

ServiceEmbeddings = {
    'azure': OpenAIEmbeddings,
}

ServiceVectors = {
    'azure': AzureCogSearch,
}

class VectorStoreFactory:
    @staticmethod
    def create(settings: IVectorStoreConfig, llm_settings: ILLMConfig) -> VectorStore:
        embedding = ServiceEmbeddings[settings['type']](
            azureOpenAIApiVersion=llm_settings.api_version,
            azureOpenAIApiKey=llm_settings.api_key,
            azureOpenAIApiInstanceName=llm_settings.instance,
            azureOpenAIApiDeploymentName=llm_settings.model
        )

        service = ServiceVectors[settings['type']](embedding, settings)

        return service
