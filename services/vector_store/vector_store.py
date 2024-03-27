from langchain_core.vectorstores import VectorStore

from interface.settings import VECTOR_STORE_TYPE, IVectorSearch
from services.vector_store.aws import AwsVectorSearch
from services.vector_store.azure import VectorAzureSearch


class VectorStoreFactory:

  @staticmethod
  def build(config: IVectorSearch) -> VectorStore:
    if config.get('type') == VECTOR_STORE_TYPE.azure_search:
      vector_search = VectorAzureSearch(config)
    elif config.get('type') == VECTOR_STORE_TYPE.open_search:
      vector_search = AwsVectorSearch(config)
    else:
      raise Exception("Invalid vector search type")
    return vector_search.build()
