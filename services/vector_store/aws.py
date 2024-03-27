import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.opensearch_vector_search import \
    OpenSearchVectorSearch
from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection

from interface.settings import IVectorSearch


class AwsVectorSearch:

  def __init__(self, config: IVectorSearch) -> None:
    self.config = config
    self.client = boto3.client(
      's3',
      aws_access_key_id=config.get('api_key'),
      aws_secret_acess_key=config.get('secret_access_key'),
      aws_session_token=config.get('session_token')
    )

  def get_embedding(self):
    config = self.config
    embedding = config.get('embedding')
    return BedrockEmbeddings(model_id=embedding.get('model_deployment'), client=self.client)

  def build(self):
    config = self.config
    bedrock_embeddings = self.get_embedding()
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
        embedding_function=bedrock_embeddings,
        opensearch_url=config.get('endpoint'),
        http_auth=auth,
        timeout=100,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
