from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ai_enterprise_agent.interface.settings import VECTOR_STORE_TYPE, ISettings
from ai_enterprise_agent.services.vector_store.aws import AwsVectorSearch
from ai_enterprise_agent.services.vector_store.azure import AzureVectorSearch
from ai_enterprise_agent.services.vector_store.pinecone import \
    PineconeVectorSearch


class VectorStoreFactory:

  @staticmethod
  def build(config: ISettings, model: BaseChatModel) -> VectorStore:
    vector_store_config = config.get('vector_store')
    if vector_store_config.get('type') == VECTOR_STORE_TYPE.azure_search:
      vector_search = AzureVectorSearch(config, model)
    elif vector_store_config.get('type') == VECTOR_STORE_TYPE.open_search:
      vector_search = AwsVectorSearch(config, model)
    elif vector_store_config.get('type') == VECTOR_STORE_TYPE.pinecone:
      vector_search = PineconeVectorSearch(config, model)
    else:
      raise Exception("Invalid vector search type")
    return vector_search
