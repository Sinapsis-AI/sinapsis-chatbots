<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Chatbots Base
<br>
</h1>

<h4 align="center">Package with base support for chat completion tasks </h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#example">📚 Usage example</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-chatbots-base` module provides core functionality for llm chat completion tasks
<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-chatbots-base --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-chatbots-base --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-chatbots-base[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-chatbots-base[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">LLMTextCompletionBase</span></strong></summary>

Base class for all templates intended to perform chat (text) completion tasks.

These attributes apply to `LLMTextCompletionBase`:
- `init_args`(`LLMInitArgs`, required): Base model arguments, including the 'llm_model_name'.
- `completion_args`(`LLMCompletionArgs`, optional): Base generation arguments, including 'max_tokens', 'temperature', 'top_p', and 'top_k'.
- `chat_history_key`(`str`, optional): Key in the packet's generic_data to find
the conversation history.
- `rag_context_key`(`str`, optional): Key in the packet's generic_data to find
RAG context to inject.
- `system_prompt`(`str | Path`, optional): The system prompt (or path to one)
to instruct the model.
- `pattern`(`dict`, optional): A regex pattern used to post-process the model's response.
- `keep_before`(`bool`, optional): If True, keeps text before the 'pattern' match; otherwise, keeps text after.

</details>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">QueryContextualize</span></strong></summary>

A base class for contextualizing queries based on certain keywords.

These attributes apply to `QueryContextualize`:
- `keywords`(`list[str]`, required): A list of keywords to be used for retrieving context.

</details>

<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">QueryContextualizeFromFile</span></strong></summary>

A subclass of QueryContextualize that retrieves context from files loaded into the `generic_data`.

These attributes apply to `QueryContextualizeFromFile`:
- `keywords`(`list[str]`, required): A list of keywords to be used for retrieving context.
- `generic_keys`(`list[str]`, required): A list of keywords that can be used to retrieve specific context data from the `generic_data` of the container.

</details>

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Data Tools.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***QueryContextualizeFromFile*** use ```sinapsis info --example-template-config QueryContextualizeFromFile``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: QueryContextualizeFromFile
  class_name: QueryContextualizeFromFile
  template_input: InputTemplate
  attributes:
    keywords: '`replace_me:list[str]`'
    generic_keys: '`replace_me:list[str]`'
```

<h2 id="example">📚 Usage example</h2>
The following agent passes a text message through a TextPacket and checks if there is context with the chosen keyword
<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Config</span></strong></summary>

```yaml
agent:
  name: chat_completion
  description: Agent with a chatbot that makes a call to the LLM model using a context uploaded from a file

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: PyPDFLoaderWrapper
  class_name: PyPDFLoaderWrapper
  template_input: InputTemplate
  attributes:
    add_document_as_text_packet: false
    pypdfloader_init:
      file_path: '/path/to/a/file.pdf'

- template_name: TextInput
  class_name: TextInput
  template_input: PyPDFLoaderWrapper
  attributes:
    text: what is AI?

- template_name: QueryContextualizeFromFile
  class_name: QueryContextualizeFromFile
  template_input: TextInput
  attributes:
    keywords: 'Artificial Intelligence'
    generic_keys: 'PyPDFLoaderWrapper'
```

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.





