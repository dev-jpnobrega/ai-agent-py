import os
import traceback
from typing import List

from langchain_community.document_loaders.text import TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ai_enterprise_agent.interface.settings import ISettings

from .loader import Loader


class TxtLoader():
  def __init__(self, config: ISettings):
    resource = Loader.build(config)
    self.vector_store: VectorStore = resource.get('vector_store')
    self.embeddings: Embeddings = resource.get('embeddings')

  async def load_file(self, file: str, file_name: str, chat_uid:str='global', tags:str = None) -> List[str]:
    try:
      with open(file, "r") as f:
        file_content = f.read().encode('utf-8')
      documents = self.split_file(file_content)
      indexed_documents = Loader.index_documents(file_name, documents, chat_uid, tags, embeddings=self.embeddings)
      documents = self.vector_store.add_documents(documents=indexed_documents)
      return documents
    except Exception as e:
      print(f'Error while loading file: {e}')
      traceback.print_exc()
      return []

  def split_file(self, file_content: str) -> List[str]:
    try:
      temp_file_path = Loader.create_temporally_file(file_content)

      txt_loader = TextLoader(temp_file_path)
      docs = txt_loader.load()

      os.unlink(temp_file_path)
    except Exception as e:
      traceback.print_exc()
      print('Error Exception: ', e)
      return e

    return docs
