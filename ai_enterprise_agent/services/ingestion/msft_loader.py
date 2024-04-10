import traceback
from typing import List

from langchain_community.document_loaders.doc_intelligence import \
    AzureAIDocumentIntelligenceLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ai_enterprise_agent.interface.settings import ISettings

from .loader import Loader


class MicrosoftLoader():
  def __init__(self, config: ISettings):
    resource = Loader.build(config)
    self.vector_store: VectorStore = resource.get('vector_store')
    self.embeddings: Embeddings = resource.get('embeddings')
    self.config = config.get('document_intelligence')

  async def load_file(self, file: str, file_name: str, chat_uid='global') -> List[str]:
    try:
      documents = self.split_file(file)
      indexed_documents = Loader.index_documents(file_name, documents, chat_uid, embeddings=self.embeddings)
      documents = self.vector_store.add_documents(documents=indexed_documents)
      return documents
    except Exception as e:
      print(f'Error while loading file: {e}')
      traceback.print_exc()
      return []

  def split_file(self, file_content: str) -> List[str]:
    try:
      loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=self.config.get('endpoint'),
        api_key=self.config.get('api_key'),
        file_path=file_content,
        api_model="prebuilt-read"
      )
      docs = loader.load()
    except Exception as e:
      traceback.print_exc()
      print('Error Exception: ', e)
      return e

    return docs
