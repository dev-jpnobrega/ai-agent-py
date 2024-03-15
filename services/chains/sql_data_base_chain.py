import json
import re
from typing import List

from langchain.callbacks import CallbackManagerForChainRun
from langchain.chains import BaseChain
from langchain.chains.sql_db import DEFAULT_SQL_DATABASE_PROMPT
from langchain.prompts import (BasePromptTemplate, ChatPromptTemplate,
                               HumanMessagePromptTemplate, MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.schema import ChainValues, RunnableSequence
from langchain.schema.output_parser import StringOutputParser
from langchain.schema.runnable import Runnable
from langchain.sql_db import SqlDatabase

MESSAGES_ERRORS = {
    "dataTooBig": "Data result is too big. Please, be more specific.",
    "dataEmpty": "Data result is empty. Please, be more specific.",
}

class SqlDatabaseChain(BaseChain):
    def __init__(
        self,
        fields: SqlDatabaseChainInput,
        customMessage: str = "",
    ) -> None:
        super().__init__(fields)
        self.llm = fields.llm
        self.database = fields.database
        self.topK = fields.topK or self.topK
        self.inputKey = fields.inputKey or self.inputKey
        self.outputKey = fields.outputKey or self.outputKey
        self.sqlOutputKey = fields.sqlOutputKey or self.sqlOutputKey
        self.prompt = fields.prompt or DEFAULT_SQL_DATABASE_PROMPT
        self.customMessage = customMessage

    def getSQLPrompt(self) -> str:
        return """
        Based on the SQL table schema provided below, write an SQL query that answers the user's question.\n
        Your response must only be a valid SQL query, based on the schema provided.\n
        Remember to put double quotes around database table names.\n
        -------------------------------------------\n
        Here are some important observations for generating the query:\n
        - Only execute the request on the service if the question is not in CHAT HISTORY, if the question has already been answered, use the same answer and do not make a query on the database.\n
        {user_prompt}\n
        -------------------------------------------\n
        SCHEMA: {schema}\n
        -------------------------------------------\n
        CHAT HISTORY: {format_chat_messages}\n
        -------------------------------------------\n
        QUESTION: {question}\n
        ------------------------------------------\n
        SQL QUERY:
        """

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
            codeBlocks = [match.replace("sql", "") for match in matches]
            sqlBlock = codeBlocks[0]

            return sqlBlock

        return None

    async def checkResultDatabase(self, database: SqlDatabase, sql: str) -> int:
        prepareSql = sql.replace(";", "")
        prepareCount = f"SELECT COUNT(*) as resultCount FROM ({prepareSql}) as tableCount;"

        try:
            countResult = await database.run(prepareCount)
            data = json.loads(countResult)
            result = int(data[0].get("resultcount", 0))

            if result >= self.maxDataExamples:
                raise Exception(MESSAGES_ERRORS["dataTooBig"])

            return result
        except Exception as error:
            raise error

    def buildPromptTemplate(self, systemMessages: str) -> BasePromptTemplate:
        combine_messages = [
            SystemMessagePromptTemplate.fromTemplate(systemMessages),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.fromTemplate("{question}"),
        ]
        CHAT_COMBINE_PROMPT = ChatPromptTemplate.fromPromptMessages(combine_messages)
        return CHAT_COMBINE_PROMPT

    async def _call(
        self, values: ChainValues, runManager: CallbackManagerForChainRun = None
    ) -> ChainValues:
        question = values[self.inputKey]
        table_schema = await self.database.getTableInfo()

        sqlQueryChain = {"schema": lambda: table_schema} | {"question": lambda input: input["question"]} | {"chat_history": lambda: values.get("chat_history")} | {"format_chat_messages": lambda: values.get("format_chat_messages")} | {"user_prompt": lambda: self.customMessage} | self.buildPromptTemplate(self.getSQLPrompt()) | self.llm.bind({"stop": ["\nSQLResult:"]})

        finalChain = {"question": lambda input: input["question"], "query": sqlQueryChain} | {
                "table_info": lambda: table_schema,
                "input": lambda: question,
                "schema": lambda: table_schema,
                "question": lambda input: input["question"],
                "query": lambda input: input["query"],
                "response": lambda input: self.getResponse(input["query"]),
            } | {
                self.outputKey: self.prompt
                    .pipe(self.llm)
                    .pipe(StringOutputParser()),
                self.sqlOutputKey: lambda previousStepResult: previousStepResult["query"]["content"],
            }

        result = await finalChain.invoke({"question": question})
        return result

    def _chainType(self) -> str:
        return "sql_chain"

    @property
    def inputKeys(self) -> List[str]:
        return [self.inputKey]

    @property
    def outputKeys(self) -> List[str]:
        return [self.outputKey, self.sqlOutputKey] if self.sqlOutputKey else [self.outputKey]
