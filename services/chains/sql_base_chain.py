from langchain.chains import BaseChain
from langchain.prompts import PromptTemplate
from langchain.sql_db import SqlDatabase
from langchain.chains.sql_db import SqlDatabaseChain
from langchain.chat_models.base import BaseChatModel
from interface.agent_interface import IDataSourceConfig
from .chain import IChain
from .sql_data_base_chain import SqlDataBaseChain

SYSTEM_MESSAGE_DEFAULT = (
    "Based on the table schema below, question, SQL query, and SQL response, write a natural language response:"
    "------------\n"
    "SCHEMA: {schema}\n"
    "------------\n"
    "QUESTION: {question}\n"
    "------------\n"
    "SQL QUERY: {query}\n"
    "------------\n"
    "SQLResult: {response}\n"
    "------------\n"
    "NATURAL LANGUAGE RESPONSE:"
)

class SqlChain(IChain):
    def __init__(self, settings: IDataSourceConfig) -> None:
        self._settings = settings
        self._dataSourceInstance = None

    def getSystemMessage(self) -> str:
        customize_message = self._settings.get("customizeSystemMessage", "")
        return SYSTEM_MESSAGE_DEFAULT + customize_message

    async def getDataSourceInstance(self) -> SqlDatabase:
        if self._dataSourceInstance is None:
            self._dataSourceInstance = await SqlDatabase.fromDataSourceParams(
                appDataSource=self._settings["dataSource"],
                **self._settings,
            )
        return self._dataSourceInstance

    async def create(self, llm: BaseChatModel, *args) -> BaseChain:
        database = await self.getDataSourceInstance()
        system_template = self.getSystemMessage()

        chain_sql = SqlDataBaseChain(
            llm=llm,
            database=database,
            outputKey="sqlResult",
            sqlOutputKey="sqlQuery",
            prompt=PromptTemplate(
                inputVariables=["question", "response", "schema", "query", "chat_history"],
                template=system_template,
            ),
            customMessage=self._settings.get("customizeSystemMessage"),
        )

        return chain_sql
