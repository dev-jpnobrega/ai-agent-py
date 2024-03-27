from typing import Any, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from interface.settings import ISettings
from interface.vector_search import ISearchType
from services.vector_store.vector_store import VectorStoreFactory


class VectorStoreChain(VectorStore):

  def __init__(self, config: ISettings):
    self._vector_store = VectorStoreFactory.build(config.get('vector_store'))

  def similarity_search(self, query: str, k: int = 4, search_type: ISearchType = ISearchType.similarity) -> List[Document]:
    docs = self._vector_store.similarity_search(
      query=query,
      k=k,
      search_type=search_type
    )
    return docs

  def similarity_search_with_relevance_scores(self, query: str,  score_threshold: float, k: Optional[int]) -> List[Tuple[Document, float]]:
    docs_and_scores = self._vector_store.similarity_search_with_relevance_scores(
      query=query,
      k=k,
      score_threshold=score_threshold,
    )
    return docs_and_scores

  def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
    return self._vector_store.search(query, search_type, **kwargs)

  def add_documents(self, documents: List[Document]):
    self._vector_store.add_documents(documents=documents)

  def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
    pass

  def from_texts(
        cls: Type[Any],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ):
    pass
