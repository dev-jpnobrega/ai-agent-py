import boto3
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from ai_enterprise_agent.interface.settings import (LLM_TYPE,
                                                    VECTOR_STORE_TYPE,
                                                    IEmbedding, ISettings)


class AzureEmbedding:

  @staticmethod
  def build(config: IEmbedding) -> Embeddings:
    return AzureOpenAIEmbeddings(
      azure_deployment=config.get('model_deployment'),
      api_version=config.get('api_version'),
      azure_endpoint=config.get('endpoint'),
      api_key=config.get('api_key')
    )

class BedrockEmbedding:

  @staticmethod
  def build(config: IEmbedding) -> Embeddings:
    client = boto3.client(
      's3',
      aws_access_key_id=config.get('api_key'),
      aws_secret_access_key=config.get('secret_access_key'),
      aws_session_token=config.get('session_token')
    )
    embedding = config.get('embedding')
    return BedrockEmbeddings(model_id=embedding.get('model_deployment'), client=client)

class GoogleEmbedding:

  @staticmethod
  def build(config: IEmbedding) -> Embeddings:
    return GoogleGenerativeAIEmbeddings(
      azure_deployment=config.get('model_deployment'),
      api_key=config.get('api_key')
    )

class EmbeddingFactory:

  @staticmethod
  def build(type: VECTOR_STORE_TYPE, config: ISettings) -> Embeddings:
    vector_store_config = config.get('vector_store')
    if type == VECTOR_STORE_TYPE.azure_search:
        return AzureEmbedding.build(vector_store_config.get('embedding'))
    elif type == VECTOR_STORE_TYPE.open_search:
        return BedrockEmbedding.build(vector_store_config.get('embedding'))
    elif type == VECTOR_STORE_TYPE.pinecone:
        model_type = config.get('model', {}).get('type')
        if model_type == LLM_TYPE.aws:
            return BedrockEmbedding.build(vector_store_config.get('embedding'))
        elif model_type == LLM_TYPE.azure:
          return AzureEmbedding.build(vector_store_config.get('embedding'))
        elif model_type == LLM_TYPE.google:
            return GoogleEmbedding.build(vector_store_config.get('embedding'))
    raise ValueError("Invalid or unsupported embedding type")

