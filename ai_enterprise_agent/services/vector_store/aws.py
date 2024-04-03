from typing import Any, Iterable, List, Optional, Tuple, Type

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores.opensearch_vector_search import \
    OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection

from ai_enterprise_agent.interface.settings import VECTOR_STORE_TYPE, ISettings
from ai_enterprise_agent.interface.vector_search import ISearchType
from ai_enterprise_agent.services.vector_store.embedding import \
    EmbeddingFactory


class AwsVectorSearch(VectorStore):

  def __init__(self, config: ISettings, model: BaseChatModel) -> None:
    super().__init__()
    self.config = config
    self.vector_store = self.build()
    self.model = model

  def build(self):
    config = self.config.get('vector_store')
    embeddings = EmbeddingFactory.build(VECTOR_STORE_TYPE.open_search, self.config)
    auth = AWSV4SignerAuth(
      credentials={
        'aws_access_key_id': config.get('api_key'),
        'aws_secret_acess_key': config.get('secret_access_key'),
        'aws_session_token': config.get('session_token')
      },
      region=config.get('region'),
      service=config.get('service')
    )
    return OpenSearchVectorSearch(
        index_name=config.get('index_name'),
        embedding_function=embeddings,
        opensearch_url=config.get('endpoint'),
        http_auth=auth,
        timeout=100,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

  def similarity_search(self, query: str, k: int = 4, search_type: ISearchType = ISearchType.similarity) -> List[Document]:
    return self.vector_store.similarity_search(
      query=query,
      k=k,
      search_type=search_type
    )

  def similarity_search_with_relevance_scores(self, query: str,  score_threshold: float, k: Optional[int]) -> List[Tuple[Document, float]]:
    return self.vector_store.similarity_search_with_relevance_scores(
      query=query,
      k=k,
      score_threshold=score_threshold,
    )

  def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
    return self.vector_store.search(query, search_type, **kwargs)

  def _call(self, query, return_source_documents: bool = True):
    retriever = self.vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(self.model, chain_type="stuff", retriever=retriever, return_source_documents=return_source_documents)
    return qa.invoke({"query": query})

  def add_documents(self, documents: List[Document]):
    self.vector_store.add_documents(documents=documents)

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
