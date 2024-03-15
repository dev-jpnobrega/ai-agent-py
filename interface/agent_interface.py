from typing import Dict, Union, List, Optional
from enum import Enum
from sqlalchemy.engine import Connection

SYSTEM_MESSAGE_DEFAULT = """
You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) or javascript (in a javascript coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
"""
class DATABASE_TYPE(Enum):
    cosmos = 'cosmos'
    redis = 'redis'
    postgres = 'postgres'

class LLM_TYPE(Enum):
    azure = 'azure'
    aws = 'aws'
    google = 'google'
    gpt = 'gpt'

class IDatabaseConfig:
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

class IDataSourceConfig:
    data_source: Connection
    includes_tables: Optional[List[str]]
    ignore_tables: Optional[List[str]]
    customize_system_message: Optional[str]
    ssl: bool = False

class IOpenAPIConfig:
    data: str
    customize_system_message: Optional[str]
    x_api_key: Optional[str]
    authorization: Optional[str]
    timeout: Optional[int]

class IChatConfig:
    temperature: float
    top_p: Optional[float]
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    max_tokens: Optional[int]

class ILLMConfig:
    type: LLM_TYPE
    model: str
    instance: Optional[str]
    api_key: Optional[str]
    api_version: str
    secret_access_key: Optional[str]
    session_token: Optional[str]
    region: Optional[str]

class IVectorStoreConfig:
    name: str
    type: LLM_TYPE
    api_key: str
    api_version: str
    indexes: Union[List[str], str]
    vector_field_name: str
    model: Optional[str]
    custom_filters: Optional[str]

class IAgentConfig:
    name: Optional[str]
    debug: bool = False
    system_message: Union[str, type(SYSTEM_MESSAGE_DEFAULT)] = None
    llm_config: ILLMConfig
    chat_config: IChatConfig
    db_history_config: Optional[IDatabaseConfig]
    vector_store_config: Optional[IVectorStoreConfig]
    data_source_config: Optional[IDataSourceConfig]
    open_api_config: Optional[IOpenAPIConfig]

class IInputProps:
    question: Optional[str]
    user_session_id: Optional[str]
    chat_thread_id: Optional[str]

TModel: Dict[str, Union[str, int]] = {}

class IAgent:
    async def call(self, input: IInputProps) -> None:
        pass

    def emit(self, event: str, *args: any) -> None:
        pass

    def on(self, event_name: Union[str, type], listener: callable) -> 'IAgent':
        pass
