<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
sinapsis-llama-cpp
<br>
</h1>

<h4 align="center">Package with support for the llama-cpp library to handle text processing </h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#example">📚 Usage example</a> •
<a href="#webapps">🌐 Webapps</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-llama-cpp` module provides a suite of templates to run LLMs with [llama-cpp](https://github.com/ggml-org/llama.cpp).
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


Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-llama-cpp --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-llama-cpp --extra-index-url https://pypi.sinapsis.tech
```

> [!IMPORTANT]
> Templates may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:
>

with <code>uv</code>:

```bash
  uv pip install sinapsis-llama-cpp[all] --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-llama-cpp[all] --extra-index-url https://pypi.sinapsis.tech
```


<h2 id="features">🚀 Features</h2>
* LLaMATextCompletion: Configures and initializes a chat completion model, supporting LLaMA, Mistral, and other compatible models.
<details>
<summary id="configuration"><strong><span style="font-size: 1.25em;">🌍 General Attributes</span></strong></summary>

These attributes apply to `LLaMATextCompletion``
:
- `llm_model_name`(Required): Name of the LLM to use.
- `llm_model_file`(Required): File path to the LLM.
- `n_ctx`(Required): Maximum context size.
- `role`: Role in the conversation (`system`, `user`, or `assistant`, default: `assistant`)
- `system_prompt` (Optional): Defines the personality of the LLM (e.g., you are a python expert)
- `prompt`: Custom instructions to guide the LLM response (default: empty).
- `chat_format`: Chat message format (`llama-2`, `chatml`, etc., default: `chatml`).
- `context_max_len`: Maximum conversation context length (default: 6).
- `pattern`: Regex pattern to match delimiters (default: handles `<|...|>` and `</...>`).
- `keep_before`: Determines which part of the matched text to return (default: `True`)
- `max_tokens`: Maximum number of tokens to generate (default: 256).
- `temperature`: Sampling temperature, controlling randomness (default: 0.5).
- `n_threads`: Number of CPU threads to use (default: 4).
- `n_gpu_layers`: Number of LLM layers offloaded to GPU (-1 for all layers, default: 0).

</details>
> [!IMPORTANT]
> We now include support for Llama4 models!

To use them, install the dependency (if you have not installed sinapsis-llama-cpp[all])
```bash
  uv pip install sinapsis-llama-cpp[llama-four] --extra-index-url https://pypi.sinapsis.tech
```
and test it through the cli or the webapp by changing the AGENT_CONFIG_PATH


> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Data Tools.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***LlaMATextCompletion*** use ```sinapsis info --example-template-config LlaMATextCompletion``` to produce the following example config:

```yaml
agent:
  name: my_first_chatbot
  description: Agent with a template to pass a text through a LLM and return a response
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: LLaMATextCompletion
  class_name: LLaMATextCompletion
  template_input: InputTemplate
  attributes:
    llm_model_name: 'bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF'
    llm_model_file: 'DeepSeek-R1-Distill-Qwen-7B-Q5_K_S.gguf'
    n_ctx: 9000
    max_tokens: 10000
    role: assistant
    system_prompt: 'You are an AI expert'
    chat_format: chatml
    context_max_len: 6
    pattern: null
    keep_before: true
    temperature: 0.5
    n_threads: 4
    n_gpu_layers: 8
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
- template_name: LLaMATextCompletion
  class_name: LLaMATextCompletion
  template_input: TextInput
  attributes:
    llm_model_name: bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF
    llm_model_file: DeepSeek-R1-Distill-Qwen-7B-Q5_K_S.gguf
    n_ctx: 9000
    max_tokens: 10000
    temperature: 0.7
    n_threads: 8
    n_gpu_layers: 29
    chat_format: chatml
    system_prompt : "You are a python and AI agents expert and you provided reasoning behind every answer you give."
    keep_before: True
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
2. **Start the container**
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-simple-chatbot -d
```
2. Check the status:
```bash
docker logs -f sinapsis-simple-chatbot
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

**To use a different chatbot configuration (e.g. OpenAI-based chat), update the `AGENT_CONFIG_PATH` environmental variable to point to the desired YAML file.**

For example, to use OpenAI chat:
```yaml
environment:
 AGENT_CONFIG_PATH: webapps/configs/openai_simple_chat.yaml
 OPENAI_API_KEY: your_api_key
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
uv run webapps/llama_cpp_simple_chatbot.py
```

**NOTE:** To use OpenAI for the simple chatbot, set your API key and specify the correct configuration file
```bash
export AGENT_CONFIG_PATH=webapps/configs/openai_simple_chat.yaml
export OPENAI_API_KEY=your_api_key
```
and run step 5 again

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

The LLama4TextToText template is licensed under the [official Llama4 license](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)



