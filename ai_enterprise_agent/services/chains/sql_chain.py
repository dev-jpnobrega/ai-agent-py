import re
from operator import itemgetter
from typing import Any, Dict

from langchain.chains.base import Chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from sqlalchemy import URL

from ai_enterprise_agent.interface.chat_history import IChatHistoryService
from ai_enterprise_agent.interface.settings import (DIALECT_TYPE,
                                                    PROCESSING_TYPE, ISettings)


class SqlChain(Chain):

  input_key = 'question'
  output_key = 'sql_chain'
  model: BaseChatModel = None
  memory: IChatHistoryService = None
  db: SQLDatabase = None
  config: ISettings = None

  @property
  def input_keys(self):
    return [self.input_key]

  @property
  def output_keys(self):
    return [self.output_key]

  def __init__(self, config: ISettings,  model: BaseChatModel, memory: IChatHistoryService) -> None:
    super().__init__()
    self.model = model
    self.memory = memory
    self.config = config
    self.db = SQLDatabase.from_uri(database_uri=self.build_uri_connect())

  def build_uri_connect(self) -> str:
    config = self.config
    database = config.get('database')
    if database.get('type') == DIALECT_TYPE.postgres:
      return URL.create(
        "postgresql+psycopg2",
        username=database.get('username'),
        password=database.get('password'),
        host=database.get('host'),
        database=database.get('database'),
      )
    else:
      raise Exception("Invalid database connection")

  def get_schema(self, _):
    config = self.config
    database = config.get('database')
    try:
      return self.db.get_table_info(database.get('includes_tables'))
    except Exception as e:
      print(f"Error executing query: {e}")

  def run_query(self, query):
    try:
      return self.db.run(query)
    except Exception as e:
      print(f"Error executing query: {e}")

  def create_sql(self, _):
    template = """Based on the table schema below, write only a SQL query that would answer the user's question.
    Your response must only be a valid SQL query, based on the schema provided.
    Remember to put double quotes around database table names. Remove triple quotes and sql word of the sentences.
    Here are some important observations for generating the query:
    - Only execute the request on the service if the question is not in History, if the question has already been answered, use the same answer and do not make a query on the database.
    If you don't find out what the table schema is, only response friendly can't be written a valid SQL query.
    Schema: {schema}
    History: {history}
    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_template(template)
    return (
      RunnablePassthrough.assign(history=RunnableLambda(self.memory.get_messages()) | itemgetter("history"))
        .assign(schema=self.get_schema)
      | prompt
      | self.model.bind(stop=["\nSQLResult:"])
      | StrOutputParser()
    )

  def parserSQL(self, sql: str) -> str:
    sqlL = sql.lower()
    if (
        sqlL.startswith("select")
        or sqlL.startswith("update")
        or sqlL.startswith("delete")
        or sqlL.startswith("insert")
    ):
      return sql

    if "```sql" in sqlL:
      regex = r"```(.*?)```"
      matches = re.findall(regex, sqlL, re.DOTALL)
      code_blocks = [match.replace("sql", "") for match in matches]
      sql_block = code_blocks[0]
      return sql_block
    return None

  def chain(self, custom_system_message) -> RunnableSerializable[Any, Any]:
    template = """If you don't get a valid query to execute, only reply in a friendly manner that you didn't find the answer.\n
    Only execute the request on the service if the question is not in History, if the question has already been answered, use the same answer and do not make a query on the database.\n
    Based on the table schema below, question, sql query, and sql response, write a natural language response:\n\n
    {custom_system_message}
    Schema: {schema}
    History: {history}
    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""
    prompt = ChatPromptTemplate.from_template(template)
    try:
      chain = (
        RunnablePassthrough.assign(history=RunnableLambda(self.memory.get_messages()) | itemgetter("history"))
          .assign(custom_system_message=lambda _: custom_system_message)
          .assign(schema=self.get_schema)
          .assign(query=self.create_sql)
          .assign(response=lambda x: self.db.run(self.parserSQL(x["query"])) if self.parserSQL(x["query"]) != None else x["query"])
        | prompt | self.model | StrOutputParser()
      )
      return chain
    except Exception as e:
      print(f"Error: {e}")

  async def _call(self, input: Dict[str, Any] = None):
    custom_system_message = input.get('custom_system_message', None)
    chain = self.chain(custom_system_message)
    response = chain.invoke(input)
    if self.config.get('processing_type') == PROCESSING_TYPE.sequential:
      return { self.output_key: response }
    return response

  @property
  def _chain_type(self) -> str:
    return "sql_chain"
