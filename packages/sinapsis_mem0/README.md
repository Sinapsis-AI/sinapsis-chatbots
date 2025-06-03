<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Mem0
<br>
</h1>

<h4 align="center">Persistent memory for AI agents: store, retrieve, and manage context across conversations and workflows.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#example">📚 Usage example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-mem0` module adds short and long-term memory to AI agents, enabling dynamic context recall, personalized interactions, and seamless knowledge retention across sessions.
<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-mem0 --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-mem0 --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-mem0[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-mem0[all] --extra-index-url https://pypi.sinapsis.tech
```


<h2 id="features">🚀 Features</h2>

The templates include the possibility to extend the AI agents with memory persistence using `Memory` or `MemoryClient`. Using the client version requires setting the API Key as:

```bash
export MEM0_API_KEY="your-api-key-here"
```

You can check the documentation for the [managed client](https://docs.mem0.ai/api-reference) version or for the [self hosted open-source](https://docs.mem0.ai/open-source/python-quickstart) version for more details on how to set them up.

<h3> Templates Supported</h3>

* Mem0Add: Ingests and structures AI interactions (prompts or responses) into Mem0’s memory format. Supports both discrete facts and full conversation histories.
* Mem0Delete: Removes memories, either selectively (by ID) or in bulk (e.g., all data for a user/agent).
* Mem0Get: Retrieves stored memories, from individual entries to entire conversation histories.
* Mem0Reset: Clears memory storage completely or within a defined scope (user/agent/run).
* Mem0Search: Dynamically fetches and injects relevant memories into prompts, with configurable formatting.

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">🌍 General Attributes</span></strong></summary>

These attributes apply to all the available templates:
- `use_managed`(`bool`, required): If True, use the managed Mem0 API (`MemoryClient`), else the self-hosted infrastructure will be used through the `Memory` class.
- `memory_config`(`dict`, optional): Parameters to configure either `MemoryClient` or `Memory`, depending on the value of `use_managed`.

</details>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">➕ Mem0Add Attributes</span></strong></summary>

- `add_kwargs`(`dict`, optional): Dictionary of parameters to pass to `MemoryClient` or `Memory` `add` method. Common keys include `user_id` or `agent_id`.
- `generic_key`(`str`, required): Key used to retrieve original prompts from the container's
generic data field.

</details>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">🗑️ Mem0Delete Attributes</span></strong></summary>

- `delete_all`(`bool`, optional): If True, performs a complete memory wipe for the specified scope (agent, run or user). If False, performs targeted deletion based on `memory_id`. Defaults to `False`.
- `delete_kwargs`(`dict`, required): Parameters for the deletion operation. Depending on the type of deletion, this may include `user_id`, `agent_id`, or `memory_id`.

</details>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">📥 Mem0Get Attributes</span></strong></summary>

- `get_all`(`bool`, optional): If True, retrieves all memories for the given context (e.g., user, agent, or run). If False, retrieves a specific memory using parameters like `memory_id`. Defaults to `False`.
- `get_kwargs`(`dict`, required): Additional parameters to pass to the memory retrieval method. Can include fields such as `user_id`, `agent_id`, or `memory_id`, depending on the attributes chosen.

</details>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">🔎 Mem0Search Attributes</span></strong></summary>

- `enclosure`(`Literal["plain", "bracket", "dashed", "xml"]`, optional): Determines how relevant memories are injected into the prompt before the user query. Defaults to `plain` which injects no special section or title, just memories and query.
- `search_kwargs`(`dict`, required): Additional parameters to pass to the memory search method. Can include fields such as `user_id`, `top_k`, or `threshold`, depending on the attributes chosen.

</details>

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
- template_name: Mem0Search
  class_name: Mem0Search
  template_input: InputTemplate
  attributes:
    use_managed: false
    memory_config: '`replace_me:dict[str, typing.Any]`'
    search_kwargs: '`replace_me:dict[str, typing.Any]`'
    enclosure: plain
```

<h2 id="example">📚 Usage example</h2>

The following agent retrieves all memories for the given `user_id` and `run_id` from the Mem0 Platform.
<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Config</span></strong></summary>

```yaml
agent:
  name: retriever_agent
  description: Agent that retrieves all memories from Mem0 platform.

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: Mem0Get
  class_name: Mem0Get
  template_input: InputTemplate
  attributes:
    use_managed: true
    memory_config:
      host: null
      org_id: null
      project_id: null
    get_all: true
    get_kwargs:
      user_id: my_user
      run_id: test
```
</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.





