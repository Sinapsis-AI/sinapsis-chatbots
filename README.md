<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
sinapsis-chatbots
<br>
</h1>

<h4 align="center">A comprehensive monorepo for building and deploying AI-driven chatbots with support for multiple large language models</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#packages">📦 Packages</a> •
<a href="#webapps">🌐 Webapps</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-chatbots` module is a powerful toolkit designed to simplify the development of AI-driven chatbots and Retrieval-Augmented Generation (RAG) systems. It provides ready-to-use templates and utilities for configuring and running large language model (LLM) applications, enabling developers to integrate a wide range of LLM models with ease for natural, intelligent interactions.


> [!IMPORTANT]
> We now include support for Llama4 models!

To use them, install the dependency (if you have not installed sinapsis-llama-cpp[all])
```bash
uv pip install sinapsis-llama-cpp[llama-four] --extra-index-url https://pypi.sinapsis.tech
```
You need a HuggingFace token. See the [official instructions](https://huggingface.co/docs/hub/security-tokens)
and set it using
```bash
export HF_TOKEN=<token-provided-by-hf>
```

and test it through the cli or the webapp by changing the AGENT_CONFIG_PATH

> [!NOTE]
> Llama 4 requires large GPUs to run the models.
> Nonetheless, running on smaller consumer-grade GPUs is possible, although a single inference may take hours
>


<h2 id="installation">🐍 Installation</h2>

This mono repo includes packages for AI-driven chatbots using various LLM frameworks through:
* <code>sinapsis-anthropic</code>
* <code>sinapsis-chatbots-base</code>
* <code>sinapsis-llama-cpp</code>
* <code>sinapsis-llama-index</code>
* <code>sinapsis-mem0</code>


Install using your preferred package manager. We strongly recommend using <code>uv</code>. To install <code>uv</code>, refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

Install with <code>uv</code>:

```bash
uv pip install sinapsis-llama-cpp --extra-index-url https://pypi.sinapsis.tech
```
Or with raw <code>pip</code>:
```bash
pip install sinapsis-llama-cpp --extra-index-url https://pypi.sinapsis.tech
```
**Replace `sinapsis-llama-cpp` with the name of the package you intend to install**.

> [!IMPORTANT]
> Templates in each package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

With <code>uv</code>:

```bash
uv pip install sinapsis-llama-cpp[all] --extra-index-url https://pypi.sinapsis.tech
```
Or with raw <code>pip</code>:
```bash
pip install sinapsis-llama-cpp[all] --extra-index-url https://pypi.sinapsis.tech
```

**Be sure to substitute `sinapsis-llama-cpp`  with the appropriate package name**.

> [!TIP]
> You can also install all the packages within this project:
>
```bash
uv pip install sinapsis-chatbots[all] --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="packages">📦 Packages</h2>

This repository is structured into modular packages, each facilitating the integration of AI-driven chatbots with various LLM frameworks. These packages provide flexible and easy-to-use templates for building and deploying chatbot solutions. Below is an overview of the available packages:


<details>
<summary id="anthropic"><strong><span style="font-size: 1.4em;"> Sinapsis Anthropic </span></strong></summary>

This package offers a suite of templates and utilities for building **text-to-text** and **image-to-text** conversational chatbots using [Anthropic's Claude](https://docs.anthropic.com/en/docs/overview) models.

- **AnthropicTextGeneration**: Template for text and code generation with Claude models using the Anthropic API.

- **AnthropicMultiModal**: Template for multimodal chat processing using Anthropic's Claude models.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-chatbots/blob/main/packages/sinapsis_anthropic/README.md).

</details>

<details>
<summary id="base"><strong><span style="font-size: 1.4em;"> Sinapsis Chatbots Base </span></strong></summary>

This package provides core functionality for LLM chat completion tasks.

- **QueryContextualizeFromFile**: Template that adds a certain context to the query searching for keywords in the Documents added in the generic_data field of the DataContainer

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-chatbots/blob/main/packages/sinapsis_chatbots_base/README.md).

</details>

<details>
<summary id="llama-cpp"><strong><span style="font-size: 1.4em;"> Sinapsis llama-cpp </span></strong></summary>

This package offers a suite of templates and utilities for running LLMs using [llama-cpp](https://github.com/ggml-org/llama.cpp).

- **LLama4MultiModal**: Template for multi modal chat processing using the LLama 4 model.

- **LLaMATextCompletionWithContext**: Template to initialize a LLaMA-based text completion model with context added in the prompt.

- **LLaMATextCompletion**: Configures and initializes a chat completion model, supporting LLaMA, Mistral, and other compatible models.

- **LLama4TextToText**: Template for text-to-text chat processing using the LLama 4 model.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-chatbots/blob/main/packages/sinapsis_llama_cpp/README.md).

</details>

<details>
<summary id="llama-cpp"><strong><span style="font-size: 1.4em;"> Sinapsis llama-cpp </span></strong></summary>

Package with support for various llama-index modules for text completion. This includes making calls to llms, processing and generating embeddings and Nodes, etc.

- **CodeEmbeddingNodeGenerator**: Template to generate nodes for a code base.

- **EmbeddingNodeGenerator**: Template for generating text embeddings using the HuggingFace model.

- **LLaMAIndexInsertNodes**: Template for inserting embeddings (nodes) into a PostgreSQL vector database using
    the LlamaIndex `PGVectorStore` to store vectorized data.

- **LLaMAIndexNodeRetriever**: Template for retrieving nodes from a database using embeddings.

- **LLaMAIndexRAGTextCompletion**: Template for configuring and initializing a LLaMA-based Retrieval-Augmented Generation (RAG) system.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-chatbots/blob/main/packages/sinapsis_llama_index/README.md).

</details>

<details>
<summary id="mem0"><strong><span style="font-size: 1.4em;"> Sinapsis Mem0 </span></strong></summary>

This package provides persistent memory functionality for Sinapsis agents using [Mem0](https://docs.mem0.ai/), supporting both **managed (Mem0 platform)** and **self-hosted** backends.

- **Mem0Add**: Ingests and stores prompts, responses, and facts into memory.
- **Mem0Get**: Retrieves individual or grouped memory records.
- **Mem0Search**: Fetches relevant memories and injects them into the current prompt.
- **Mem0Delete**: Removes stored memories selectively or in bulk.
- **Mem0Reset**: Fully clears memory within a defined scope.

For specific instructions and further details, see the [README.md](https://github.com/Sinapsis-AI/sinapsis-chatbots/blob/main/packages/sinapsis_mem0/README.md).

</details>

<h2 id="webapps">🌐 Webapps</h2>

The webapps included in this project showcase the modularity of the templates, in this case for AI-driven chatbots.

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-chatbots.git
cd sinapsis-chatbots
```

> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!IMPORTANT]
> You can change the model name and the number of gpu_layers used by the model in case you have an Out of Memory (OOM) error.

> [!IMPORTANT]
> Anthropic requires an API key to interact with the API. To get started, visit the [official website](https://console.anthropic.com/) to create an account. If you already have an account, go to the [API keys page](https://console.anthropic.com/settings/keys) to generate a token.

> [!IMPORTANT]
> Set your API key env var using <code> export ANTHROPIC_API_KEY='your-api-key'</code>

> [!NOTE]
> Agent configuration can be changed through the `AGENT_CONFIG_PATH` env var. You can check the available configurations in each package configs folder.

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT**: This Docker image depends on the `sinapsis-nvidia:base` image. For detailed instructions, please refer to the [Sinapsis README](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker).

1. **Build the sinapsis-chatbots image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the app container**

- For Anthropic text-to-text chatbot:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-claude-chatbot -d
```

- For llama-cpp text-to-text chatbot:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-simple-chatbot -d
```

- For llama-index RAG chatbot:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-rag-chatbot -d
```

3. **Check the logs**

- For Anthropic text-to-text chatbot:
```bash
docker logs -f sinapsis-claude-chatbot
```

- For llama-cpp text-to-text chatbot:
```bash
docker logs -f sinapsis-simple-chatbot
```

- For llama-index RAG chatbot:
```bash
docker logs -f sinapsis-rag-chatbot
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

**To use a different chatbot configuration (e.g. OpenAI-based chat), update the `AGENT_CONFIG_PATH` environmental variable to point to the desired YAML file.**

For example, to use OpenAI chat:
```yaml
environment:
 AGENT_CONFIG_PATH: webapps/configs/openai_simple_chat.yaml
 OPENAI_API_KEY: your_api_key
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

5. **Run the webapp**:

- For Anthropic text-to-text chatbot:
```bash
export ANTHROPIC_API_KEY=your_api_key
uv run webapps/claude_chatbot.py
```

- For llama-cpp text-to-text chatbot:
```bash
uv run webapps/llama_cpp_simple_chatbot.py
```

- For OpenAI text-to-text chatbot:
```bash
export AGENT_CONFIG_PATH=webapps/configs/openai_simple_chat.yaml
export OPENAI_API_KEY=your_api_key
uv run webapps/llama_cpp_simple_chatbot.py
```

- For llama-index RAG chatbot:
```bash
uv run webapps/llama_index_rag_chatbot.py
```

6. **The terminal will display the URL to access the webapp, e.g.**:


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





