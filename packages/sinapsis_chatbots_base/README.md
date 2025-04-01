<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
sinapsis-chatbots-base
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
* LLMTextCompletionBase: Base class for all templates intended to perform chat (text) completion tasks
<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">🌍 General Attributes</span></strong></summary>

These attributes apply to `LLMTextCompletionBase``:
- `llm_model_name`(Required): Name of the LLM to use.
- `n_ctx`(Required): Maximum context size.
- `role`: Role in the conversation (`system`, `user`, or `assistant`, default: `assistant`)
- `system_prompt` (Optional): Defines the personality of the LLM (e.g., you are a python expert)
- `prompt`: Custom instructions to guide the LLM response (default: empty).
- `chat_format`: Chat message format (`llama-2`, `chatml`, etc., default: `chatml`).
- `context_max_len`: Maximum conversation context length (default: 6).
- `pattern`: Regex pattern to match delimiters (default: handles `<|...|>` and `</...>`).
- `keep_before`: Determines which part of the matched text to return (default: `True`)


</details>
* QueryContextualizeFromFile: Template that adds a certain context to the query searching for keywords in the Documents
added in the generic_data field of the DataContainer

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Data Tools.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***QueryContextualizeFromFile*** use ```sinapsis info --example-template-config QueryContextualizeFromFile``` to produce the following example config:

```yaml
agent:
  name: query_contextualize_template
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
  attributes: { }
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
    generic_keys: 'PyPDFLoaderWrapper

```


<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.





