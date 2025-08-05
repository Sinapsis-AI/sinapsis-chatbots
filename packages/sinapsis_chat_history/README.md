<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Chat History
<br>
</h1>

<h4 align="center">Package with templates to store in sql databases the history of AI agents: store, retrieve, and manage context across conversations.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#example">📚 Usage example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-chat-history` module functionality to handle history and context, saving conversations in sql-databases.
<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-chat-history --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-chat-history --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-chat-history[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-chat-history[all] --extra-index-url https://pypi.sinapsis.tech
```


<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

* ChatHistoryFetcher: Template for retrieving chat histories from a storage backend.
* ChatHistoryRemover: Template for deleting chat history records based on filters.
* ChatHistoryReset: Performs complete reset of chat history by dropping and recreating the table.
* ChatHistorySaver: Template for saving chat messages into the database.




> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Mem0.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***Mem0Search*** use ```sinapsis info --example-template-config Mem0Search``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: TextInput
  class_name: TextInput
  template_input: InputTemplate
  attributes:
    text: "this is a test message"
    id: "sinapsis-user"
    source: "1"
- template_name : ChatHistorySaver
  class_name: ChatHistorySaver
  template_input: TextInput
  attributes:
    db_config:
      db_name: "test"
      table: "chat_messages2"
```

<h2 id="example">📚 Usage example</h2>

The following agent retrieves all entries in the database for the given `user_id` and `session_id`
<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Config</span></strong></summary>

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: ChatHistoryRemover
  class_name: ChatHistoryRemover
  template_input: InputTemplate
  attributes:
    provider: postgres
    db_config:
      db_name: "test"
      table: "chat_messages"
    last_n: 10
    filters:
      user_id: Chatbot user
      session_id: 3aa8d1a4-c8a7-4367-8b20-168351f76ab9

```
</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.





