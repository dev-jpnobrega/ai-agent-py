from enum import Enum
from typing import List, Optional, Union


class LLM_TYPE(Enum):
    azure = 'azure'
    aws = 'aws'
    google = 'google'
    gpt = 'gpt'

class VECTOR_STORE_TYPE(Enum):
    azure_search = 'azure_search'
    open_search = 'open_search'
    vertex_search = 'vertex_search'
    pinecone = 'pinecone'

class DIALECT_TYPE(Enum):
    postgres = 'postgres'
    sqlite = 'sqlite'
    sqlServer = 'sqlServer'
    mysql = 'mysql'

class DATABASE_TYPE(Enum):
    cosmos = 'cosmos'
    redis = 'redis'
    postgres = 'postgres'

class CHAIN_TYPE(Enum):
    simple_chain = 'simple_chain'
    open_api_chain = 'open_api_chain'
    sql_chain = 'sql_chain'
    vector_store_chain = 'vector_store_chain'

class PROCESSING_TYPE(Enum):
    single = 'single'
    sequential = 'sequential'
    orchestrated = 'orchestrated'

class DOCUMENT_INTELLIGENCE_TYPE(Enum):
    azure = 'azure'

class IModel:
    type: LLM_TYPE
    model: str
    api_key: str
    temperature: Optional[float]
    endpoint: Optional[str]
    api_version: Optional[str]
    secret_access_key: Optional[str]
    session_token: Optional[str]
    region: Optional[str]

class IDatabase:
    type: DIALECT_TYPE
    url: Optional[str]
    username = Optional[str]
    password = Optional[str]
    host = Optional[str]
    port = Optional[str]
    database = Optional[str]
    includes_tables = Optional[str]
    custom_system_message: Optional[str]

class IOpenApi:
    data: str
    token: Optional[str]
    allow_dangerous_requests: bool
    custom_system_message: Optional[str]

class IEmbedding:
    model_deployment: Optional[str]
    api_version: Optional[str]
    endpoint: Optional[str]
    api_key: Optional[str]

class IVectorSearch:
    type: VECTOR_STORE_TYPE
    endpoint: Optional[str]
    api_key: Optional[str]
    index_name: Optional[str]
    secret_access_key: Optional[str]
    session_token: Optional[str]
    embedding: IEmbedding
    region: Optional[str]
    service: Optional[str]
    custom_filters: Optional[str]
    instance: Optional[str]
    fields_content: Optional[str]
    fields_content_vector: Optional[str]
    vector_search_profile_name: Optional[str]
    cloud: Optional[str]

class IChatHistory:
    type: DATABASE_TYPE
    host: str
    port: int
    ssl: bool = False
    session_id: Optional[str]
    session_ttl: Optional[int]
    username: Optional[str]
    password: Optional[str]
    database: Optional[Union[str, int]]
    container: Optional[str]
    synchronize: bool = False
    limit: Optional[int]

class ISystem:
    system_message: str

class IDocumentIntelligence:
    type: DOCUMENT_INTELLIGENCE_TYPE
    endpoint: str
    api_key: str

class ISettings:
    model: IModel
    chains: List[CHAIN_TYPE]
    processing_type: PROCESSING_TYPE
    database: Optional[IDatabase]
    open_api: Optional[IOpenApi]
    vector_store: Optional[IVectorSearch]
    history: Optional[IChatHistory]
    document_intelligence: Optional[DOCUMENT_INTELLIGENCE_TYPE]
    system: ISystem
