
[![Publish new version to NPM](https://github.com/dev-jpnobrega/ai-agent/actions/workflows/npm-publish.yml/badge.svg)](https://github.com/dev-jpnobrega/ai-agent/actions/workflows/npm-publish.yml)

# AI Agent

AI Agent simplifies the implementation and use of generative AI with LangChain, was inspired by the project [autogen](https://github.com/microsoft/autogen)



## Installation

Use the package manager [pip](https://pypi.org/project/pip/) to install AI Agent.

```bash
pip install ai_enterprise_agent
```

## Usage

### Simple use
```python
  import asyncio

  from ai_enterprise_agent.agent import Agent
  from ai_enterprise_agent.interface.settings import (CHAIN_TYPE, DATABASE_TYPE, DIALECT_TYPE,
                                  LLM_TYPE, PROCESSING_TYPE, VECTOR_STORE_TYPE)
  agent = Agent({
    'processing_type': PROCESSING_TYPE.single,
    'chains': [CHAIN_TYPE.simple_chain],
    'model': {
      "type": LLM_TYPE.azure,
      "api_key": <api_key>,
      "model": <model>,
      "endpoint": <endpoint>,
      "api_version": <api_version>,
      "temperature": 0.0
    },
    "system": {
      "system_message": ""
    },
  })

  response = asyncio.run(
    agent._call(
      input={
        "question": "Who's Leonardo Da Vinci?.",
        "chat_thread_id": "<chat_thread_id>"
      }
    )
  )
  print(response)
```

### Using with Orchestrator Mode
When using LLM with Orchestrator Mode the Agent finds the best way to answer the question in your base knowledge.
```python

  agent = Agent({
    'processing_type': PROCESSING_TYPE.orchestrator,
    'chains': [CHAIN_TYPE.simple_chain, CHAIN_TYPE.sql_chain],
    'model': {
      "type": LLM_TYPE.azure,
      "api_key": <api_key>,
      "model": <model>,
      "endpoint": <endpoint>,
      "api_version": <api_version>,
      "temperature": 0.0
    },
     "database": {
      "type": DIALECT_TYPE.postgres,
      "host": <host>,
      "port": <port>,
      "username": <username>,
      "password": <password>,
      "database": <database>,
      "includes_tables": ['table-1', 'table-2'],
    },
    "system": {
      "system_message": ""
    },
  })

  response = asyncio.run(
    agent._call(
      input={
        "question": "How many employees there?",
        "chat_thread_id": "<chat_thread_id>"
      }
    )
  )
  print(response)
```

## Contributing

If you've ever wanted to contribute to open source, and a great cause, now is your chance!

See the [contributing docs](CONTRIBUTING.md) for more information

## Contributors ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/dev-jpnobrega">
        <img src="https://avatars1.githubusercontent.com/u/28389807?s=400&u=2c152fc946efc96badce0cfc743ebcb2585b4b3f&v=4" width="100px;" alt=""/>
        <br />
        <sub>
          <b>JP. Nobrega</b>
        </sub>
      </a>
      <br />
      <a href="https://github.com/dev-jpnobrega/ai-agent-py/issues" title="Answering Questions">💬</a>
      <a href="https://github.com/dev-jpnobrega/ai-agent-py/master#how-do-i-use" title="Documentation">📖</a>
      <a href="https://github.com/dev-jpnobrega/ai-agent-py/pulls" title="Reviewed Pull Requests">👀</a>
      <a href="#talk-kentcdodds" title="Talks">📢</a>
    </td>
    <td align="center">
      <a href="https://github.com/tuliogaio">
        <img src="https://github.com/tuliogaio.png" width="100px;" alt=""/>
        <br />
        <sub>
          <b>Túlio César Gaio</b>
        </sub>
      </a>
      <br />
      <a href="https://github.com/dev-jpnobrega/ai-agent-py/issues" title="Answering Questions">💬</a>
      <a href="https://github.com/dev-jpnobrega/ai-agent-py/master#how-do-i-use" title="Documentation">📖</a>
      <a href="https://github.com/dev-jpnobrega/ai-agent-py/pulls" title="Reviewed Pull Requests">👀</a>
      <a href="#talk-kentcdodds" title="Talks">📢</a>
    </td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## License
[Apache-2.0](LICENSE)
