<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
sinapsis-llama-index
<br>
</h1>

<h4 align="center">Package with support for the llama-index library to handle text processing </h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#example">📚 Usage example</a> •
<a href="#webapps">🌐 Webapps</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-llama-index` module provides a suite of templates to run LLMs with [llama-index](https://github.com/run-llama/llama_index).

<h2 id="installation">🐍 Installation</h2>


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-llama-index --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-llama-index --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-llama-index[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-llama-index[all] --extra-index-url https://pypi.sinapsis.tech
```


<h2 id="features">🚀 Features</h2>
* LLaMATextCompletion: Configures and initializes a chat completion model, supporting LLaMA, Mistral, and other compatible models.
* QueryContextualizeFromFile: Template that adds a certain context to the query searching for keywords in the Documents
added in the generic_data field of the DataContainer
* QueryContextualizeFromText: Template that adds a certain context to the query searching for keywords
in the ContextKeys dataclass from the helpers.
> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Data Tools.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for **CodeEmbeddingNodeGenerator** use ```sinapsis info --example-template-config CodeEmbeddingNodeGenerator``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
  description
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: CodeEmbeddingNodeGenerator
  class_name: CodeEmbeddingNodeGenerator
  template_input: InputTemplate
  attributes:
    chunk_size: '512'
    separator: ' '
    model_name: Snowflake/snowflake-arctic-embed-m-long
    trust_remote_code: false
    programming_language: python

```

<h2 id="example">📚 Usage example</h2>
The following agent passes a text message through a TextPacket and retrieves a response from a LLM
<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Config</span></strong></summary>

```yaml
agent:
  name: chat_completion
  description: Agent with a chatbot that makes a call to the LLM model using a context uploaded from a file

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: { }

- template_name: TextInput
  class_name: TextInput
  template_input: InputTemplate
  attributes:
    text: what is AI?
- template_name: EmbeddingNodeGenerator-1
  class_name: CodeEmbeddingNodeGenerator
  template_input: LLaMARAGTextCompletion
  attributes:
    chunk_size: 512
    separator: ' '
    model_name: Snowflake/snowflake-arctic-embed-m-long
    trust_remote_code: True
```
</details>
<h2 id="webapps">🌐 Webapps</h2>

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
> You can change the model name and the number of gpu_layers used by the model in case you have an Out of Memory (OOM) error


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis-nvidia:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-chatbots image**:
```bash
docker compose -f docker/compose.yaml build
```
2. **Start the POSTGRES service**:
```bash
docker compose -f docker/compose_db.yaml up --build
```
3. **Start the container**
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-rag-chatbot -d
```
4. Check the status:
```bash
docker logs -f sinapsis-rag-chatbot
```
3. The logs will display the URL to access the webapp, e.g.,:
```bash
Running on local URL:  http://127.0.0.1:7860
```
**NOTE**: The url may be different, check the logs
4. To stop the app:
```bash
docker compose -f docker/compose_apps.yaml down
```

</details>
<details>
<summary><strong><span style="font-size: 1.25em;">💻  UV</span></strong></summary>

1. Export the environment variable to install the python bindings for llama-cpp

```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE="1"
```
2. export CUDACXX:
```bash
export CUDACXX=$(command -v nvcc)
```

3. **Create the virtual environment and sync dependencies:**

```bash
uv sync --frozen
```

4. **Install the wheel**:
```bash
uv pip install sinapsis-chatbots[all] --extra-index-url https://pypi.sinapsis.tech
```

5. **Run the webapp**:
```bash
uv run webapps/llama_index_rag_chatbot.py
```

6. **The terminal will display the URL to access the webapp, e.g.**:

NOTE: The url can be different, check the output of the terminal
```bash
Running on local URL:  http://127.0.0.1:7860
```

</details>


<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.





