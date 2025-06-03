<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
sinapsis-anthropic
<br>
</h1>

<h4 align="center">Templates for text-to-text and image-to-text conversational chatbots using Anthropic's Claude models.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#example">📚 Usage example</a> •
<a href="#webapps">🌐 Webapps</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-anthropic` module provides a suite of templates for building **text-to-text** and **image-to-text** conversational chatbots using [Anthropic's Claude](https://docs.anthropic.com/en/docs/overview) models.

<h2 id="installation">🐍 Installation</h2>


Install using your preferred package manager. We strongly recommend using <code>uv</code>. To install <code>uv</code>, refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Install with <code>uv</code>:

```bash
uv pip install sinapsis-anthropic --extra-index-url https://pypi.sinapsis.tech
```

Or with raw <code>pip</code>:
```bash
pip install sinapsis-anthropic --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

With <code>uv</code>:
```bash
uv pip install sinapsis-anthropic[all] --extra-index-url https://pypi.sinapsis.tech
```
Or with raw <code>pip</code>:
```bash
pip install sinapsis-anthropic[all] --extra-index-url https://pypi.sinapsis.tech
```


<h2 id="features">🚀 Features</h2>

<h3>Templates Supported</h3>

- **AnthropicTextGeneration**: Template for text and code generation with Claude models using the Anthropic API.

    <details>
    <summary>Attributes</summary>

    - `llm_model_name`(Required): The name of the Claude model to be used. To see the list of all available Claude models visit the [official documentation](https://docs.anthropic.com/en/docs/about-claude/models/overview).
    - `role`(Optional): The role in the conversation, such as `system`, `user`, or `assistant` (default: `assistant`).
    - `prompt`(Optional):  A set of instructions provided to the LLM to guide how to respond. (default: empty string).
    - `system_prompt`(Optional): The prompt that indicates the LLM how to behave (default: `None`).
    - `context_max_len`(Optional): The maximum length for the conversation context (default: `6`).
    - `budget_tokens`(Optional): The maximum number of tokens to allocate for intermediate
        thinking steps when `extended_thinking` is enabled (default: `2000`).
    - `extended_thinking`(Optional): A flag indicating whether to display "thinking" content blocks in the response (default: `False`).
    - `max_tokens`(Optional): Maximum number of tokens to generate (default: `4000`).
    - `temperature`(Optional): Sampling temperature for the model (default: `1.0`).
    - `web_search`(Optional): A boolean flag indicating whether web search should be enabled (default: `False`).
    </details>

- **AnthropicMultiModal**: Template for multimodal chat processing using Anthropic's Claude models.

    <details>
    <summary>Attributes</summary>

    - `llm_model_name`(Required): The name of the Claude model to be used. To see the list of all available Claude models visit the [official documentation](https://docs.anthropic.com/en/docs/about-claude/models/overview).
    - `role`(Optional): The role in the conversation, such as `system`, `user`, or `assistant` (default: `assistant`).
    - `prompt`(Optional):  A set of instructions provided to the LLM to guide how to respond. (default: empty string).
    - `system_prompt`(Optional): The prompt that indicates the LLM how to behave (default: `None`).
    - `context_max_len`(Optional): The maximum length for the conversation context (default: `6`).
    - `budget_tokens`(Optional): The maximum number of tokens to allocate for intermediate
        thinking steps when `extended_thinking` is enabled (default: `2000`).
    - `extended_thinking`(Optional): A flag indicating whether to display "thinking" content blocks in the response (default: `False`).
    - `max_tokens`(Optional): Maximum number of tokens to generate (default: `4000`).
    - `temperature`(Optional): Sampling temperature for the model (default: `1.0`).
    - `web_search`(Optional): A boolean flag indicating whether web search should be enabled (default: `False`).
    </details>


> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for **AnthropicTextGeneration** use ```sinapsis info --example-template-config AnthropicTextGeneration``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: AnthropicTextGeneration
  class_name: AnthropicTextGeneration
  template_input: InputTemplate
  attributes:
    llm_model_name: claude-3-7-sonnet-latest
    n_ctx: 9000
    role: assistant
    prompt: ''
    system_prompt: null
    chat_format: chatml
    context_max_len: 6
    pattern: null
    keep_before: true
    budget_tokens: 2000
    extended_thinking: false
    max_tokens: 4000
    temperature: 1.0
    web_search: false
```

<h2 id="example">📚 Usage example</h2>

This example shows how to use the **AnthropicMultiModal** template to process both text and image inputs to generate text responses. The following agent passes a text message through a TextPacket and an image though an ImagePacket and retrieves a response from a Claude model.

<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Config</span></strong></summary>

```yaml
agent:
  name: my_claude_agent
  description: Agent with support for text-to-text and image-to-text conversational chatbots using Anthropic's Claude models

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: { }

- template_name: TextInput
  class_name: TextInput
  template_input: InputTemplate
  attributes:
    text: Describe this image in two sentences.
- template_name: ImageReader
  class_name: FolderImageDatasetCV2
  template_input: TextInput
  attributes:
    load_on_init : True
    data_dir: './artifacts'
    pattern : 'sunset.jpeg'
- template_name: AnthropicMultiModal
  class_name: AnthropicMultiModal
  template_input: ImageReader
  attributes:
    llm_model_name: claude-3-opus-20240229
    max_tokens: 4000
    temperature: 1
```
</details>

This configuration defines an **agent** and a sequence of **templates** for a multimodal conversational chatbot using an **Anthropic's Claude** model.

> [!IMPORTANT]
> The TextInput and FolderImageDatasetCV2 templates correspond to [sinapsis-data-readers](https://github.com/Sinapsis-AI/sinapsis-data-tools/tree/main/packages/sinapsis_data_readers). If you want to use the example, please make sure you install the package.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="webapps">🌐 Webapp</h2>

This module includes a webapp to interact with the model

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-chatbots.git
cd sinapsis-chatbots
```

> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!IMPORTANT]
> Anthropic requires an API key to interact with the API. To get started, visit the [official website](https://console.anthropic.com/) to create an account. If you already have an account, go to the [API keys page](https://console.anthropic.com/settings/keys) to generate a token.

> [!IMPORTANT]
> Set your API key env var using <code> export ANTHROPIC_API_KEY='your-api-key'</code>


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-chatbots image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start app the container**
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-claude-chatbot -d
```
3. **Check the logs**
```bash
docker logs -f sinapsis-claude-chatbot
```
4. **The logs will display the URL to access the webapp, e.g.,:**:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The url may be different, check the output of logs.

5. **To stop the app**:
```bash
docker compose -f docker/compose_apps.yaml down
```

</details>

<details>
<summary id="virtual-environment"><strong><span style="font-size: 1.4em;">💻 UV</span></strong></summary>

To run the webapp using the <code>uv</code> package manager, follow these steps:

1. **Export the environment variable to install the python bindings for llama-cpp**:

```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE="1"
```
2. **Export CUDACXX**:
```bash
export CUDACXX=$(command -v nvcc)
```

3. **Sync the virtual environment**:

```bash
uv sync --frozen
```

4. **Install the wheel**:
```bash
uv pip install sinapsis-chatbots[all] --extra-index-url https://pypi.sinapsis.tech
```

5. **Set your API key**:
```bash
export ANTHROPIC_API_KEY=your_api_key
```

6. **Run the webapp**:
```bash
uv run webapps/claude_chatbot.py
```

7. **The terminal will display the URL to access the webapp (e.g.)**:

```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The URL may vary; check the terminal output for the correct address.

</details>


<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.