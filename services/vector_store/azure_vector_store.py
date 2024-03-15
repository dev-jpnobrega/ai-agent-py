import aiohttp
import json
import httpx as http
from langchain.callbacks import Callbacks
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from nanoid import generate
from typing import List, Dict, Union, Tuple

AzureCogDocument = Dict[str, Union[str, int, List[int], Dict[str, any]]]
AzureCogVectorField = Dict[str, Union[List[int], str, int]]
AzureCogFilter = Dict[str, Union[str, List[str], int]]
AzureCogRequestObject = Dict[str, Union[str, List[str], int, List[AzureCogVectorField]]]
DocumentSearchResponseModel = Dict[str, List[AzureCogDocument]]

class AzureCogSearch(VectorStore):
    def __init__(self, embeddings: Embeddings, db_config: Dict[str, str]):
        embeddings.azureOpenAIApiDeploymentName = db_config['model']
        super().__init__(embeddings, db_config)
        self._config = db_config

    def _vectorstoreType(self) -> str:
        return 'azure-cog-search'

    @property
    def config(self) -> Dict[str, str]:
        return self._config

    @property
    def base_url(self) -> str:
        return f"https://{self._config['name']}.search.windows.net/indexes"

    async def addDocuments(self, documents: List[Document]) -> List[str]:
        texts = [document.pageContent for document in documents]
        vectors = await self.embeddings.embedDocuments(texts)
        return await self.addVectors(vectors, documents)

    async def similaritySearch(self, query: str, k: int = None, filter: AzureCogFilter = None) -> List[Document]:
        k = k or 4
        embeddings = await self.embeddings.embedQuery(query)
        results = await self.similaritySearchVectorWithScore(embeddings, k, filter)
        return [doc for doc, _score in results]

    async def similaritySearchWithScore(self, query: str, k: int = None, filter: AzureCogFilter = None, callbacks: Callbacks = None) -> List[Tuple[Document, int]]:
        k = k or 5
        embeddings = await self.embeddings.embedQuery(query)
        return await self.similaritySearchVectorWithScore(embeddings, k, filter, callbacks)

    async def addVectors(self, vectors: List[List[int]], documents: List[Document]) -> List[str]:
        indexes = []
        for vector, document in zip(vectors, documents):
            indexes.append({
                'id': generate().replace('_', ''),
                **document,
                self._config['vectorFieldName']: vector
            })

        for index in indexes:
            if '_' in index['id']:
                index['id'] = index['id'].replace('_', '')

        document_index_request = {'value': indexes}
        url = f"{self.base_url}/{self._config['indexes'][0]}/docs/index?api-version={self._config['apiVersion']}"
        response_obj = await fetcher(url, document_index_request, self._config['apiKey'])
        return [doc['key'] for doc in response_obj['value']]

    async def similaritySearchVectorWithScore(self, query: List[int], k: int, filter: AzureCogFilter = None, index: str = None) -> List[Tuple[Document, int]]:
        index = index or self._config['indexes'][0]
        url = f"{self.base_url}/{index}/docs/search?api-version={self._config['apiVersion']}"
        search_body = {
            'search': filter.get('search', '*'),
            'facets': filter.get('facets', []),
            'filter': filter.get('filter', ''),
            'vectors': [{'value': query, 'fields': filter.get('vectorFields', ''), 'k': k}],
            'top': filter.get('top', k)
        }
        result_documents = await fetcher(url, search_body, self._config['apiKey'])
        return [(doc, doc['@search.score'] or 0) for doc in result_documents['value']]

async def fetcher(url: str, body: Dict, api_key: str) -> Dict:
    options = {
        'method': 'POST',
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            'api-key': api_key
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, **options) as response:
            if not response.ok:
                err = await response.json()
                raise http.HTTPStatusError(json.dumps(err))
            return await response.json()
