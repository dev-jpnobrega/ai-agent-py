import hashlib
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from nanoid import generate

from ai_enterprise_agent.interface.settings import ISettings
from ai_enterprise_agent.services.llm.model import ModelFactory
from ai_enterprise_agent.services.vector_store.embedding import \
    EmbeddingFactory
from ai_enterprise_agent.services.vector_store.vector_store import \
    VectorStoreFactory


class Loader:

  @staticmethod
  def build(config: ISettings):
    llm_config = config.get('model')
    vector_store_config = config.get('vector_store')
    model = ModelFactory.build(llm_config)
    vector_store = VectorStoreFactory.build(config, model)
    embeddings = EmbeddingFactory.build(vector_store_config.get('type'), config)
    return {'vector_store': vector_store, 'embeddings': embeddings }

  @staticmethod
  def index_documents(file_name: str, documents: List[Document], chat_uid: str, embeddings: Embeddings) -> List[Document]:
    hashed_user_id = Loader.hash_user_id()
    return [Document(
      id=str(generate()),
      chatThreadId=chat_uid,
      user=hashed_user_id,
      page_content=doc.page_content,
      metadata={"file": file_name},
      embedding=embeddings.embed_query(text=doc.page_content)
    ) for doc in documents]

  @staticmethod
  def hash_user_id() -> str:
    value = 'aienterpriseagent'
    return hashlib.sha256(value.encode()).hexdigest()

  @staticmethod
  def create_temporally_file(file_content: str):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_file.write(file_content)
      temp_file_path = temp_file.name

    return temp_file_path
