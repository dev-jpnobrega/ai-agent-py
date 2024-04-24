import traceback
from typing import List

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ai_enterprise_agent.interface.settings import ISettings

from .loader import Loader


class PdfLoader():
  def __init__(self, config: ISettings):
    resource = Loader.build(config)
    self.vector_store: VectorStore = resource.get('vector_store')
    self.embeddings: Embeddings = resource.get('embeddings')

  async def load_file(self, file: str, file_name: str, chat_uid='global', tags:str = None) -> List[str]:
    try:
      documents = self.split_file(file)
      indexed_documents = Loader.index_documents(file_name, documents, chat_uid, tags, embeddings=self.embeddings)
      documents = self.vector_store.add_documents(documents=indexed_documents)
      return documents
    except Exception as e:
      print(f'Error while loading file: {e}')
      traceback.print_exc()
      return []

  def split_file(self, file_content: str) -> List[str]:
    try:
      loader = PyPDFLoader(file_content)
      docs = loader.load_and_split()
    except Exception as e:
      traceback.print_exc()
      print('Error Exception: ', e)
      return e

    return docs
