from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

from interface.settings import IVectorSearch


class VectorAzureSearch:

  def __init__(self, config: IVectorSearch) -> None:
    self._endpoint = config.get('endpoint')
    self._api_key = config.get('api_key')
    self._index_name = config.get('index_name')
    embedding = config.get('embedding')
    self._embedding = {
      'model_deployment': embedding.get('model_deployment'),
      'api_version': embedding.get('api_version'),
      'endpoint': embedding.get('endpoint'),
      'api_key': embedding.get('api_key')
    }

  def get_embedding(self):
    config = self._embedding
    embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
      azure_deployment=config.get('model_deployment'),
      openai_api_version=config.get('api_version'),
      azure_endpoint=config.get('endpoint'),
      api_key=config.get('api_key'),
    )
    return embeddings

  def build(self):
    embedding_function = self.get_embedding()
    return AzureSearch(
      azure_search_endpoint=self._endpoint,
      azure_search_key=self._api_key,
      index_name=self._index_name,
      embedding_function=embedding_function.embed_query
    )
