<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis LLaMA CPP
<br>
</h1>

<h4 align="center">Sinapsis Templates for LLM text completion with LLaMA-CPP</h4>

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

To use them, install the dependency (if you have not installed sinapsis-llama-cpp[all]):

```bash
  uv pip install sinapsis-llama-cpp[llama-four] --extra-index-url https://pypi.sinapsis.tech
```

You need a HuggingFace token. See the [official instructions](https://huggingface.co/docs/hub/security-tokens) and set it using:

```bash
  export HF_TOKEN=<token-provided-by-hf>
```

And test it through the cli or the webapp by changing the AGENT_CONFIG_PATH

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

<h3>Templates Supported</h3>

- **LLaMATextCompletion**: Template for text completion using LLaMA CPP.

    <details>
    <summary>Attributes</summary>

    - `init_args`(`LLaMAInitArgs`, required): LLaMA model arguments.
      - `llm_model_name`(`str`, required): The name or path of the LLM model to use (e.g. 'TheBloke/Llama-2-7B-GGUF').
      - `llm_model_file`(`str`, required): The specific GGUF model file (e.g., 'llama-2-7b.Q2_K.gguf').
      - `n_gpu_layers`(`int`, optional): Number of layers to offload to the GPU (-1 for all). Defaults to `0`.
      - `use_mmap`(`bool`, optional): Use 'memory-mapping' to load the model. Defaults to `True`.
      - `use_mlock`(`bool`, optional): Force the model to be kept in RAM. Defaults to `False`.
      - `seed`(`int`, optional): RNG seed for model initialization. Defaults to `LLAMA_DEFAULT_SEED`.
      - `n_ctx`(`int`, optional): The context window size. Defaults to `512`.
      - `n_batch`(`int`, optional): The batch size for prompt processing. Defaults to `512`.
      - `n_ubatch`(`int`, optional): The batch size for token generation. Defaults to `512`.
      - `n_threads`(`int`, optional): CPU threads for generation. Defaults to `None`.
      - `n_threads_batch`(`int`, optional): CPU threads for batch processing. Defaults to `None`.
      - `flash_attn`(`bool`, optional): Enable Flash Attention if supported by the GPU. Defaults to `False`.
      - `chat_format`(`str`, optional): Chat template format (e.g., 'chatml'). Defaults to `None`.
      - `verbose`(`bool`, optional): Enable verbose logging from llama.cpp. Defaults to `True`.
    - `completion_args`(`LLaMACompletionArgs`, required): Generation arguments to pass to the selected model.
      - `temperature`(`float`, optional): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
      - `top_p`(`float`, optional): Nucleus sampling. Considers tokens with cumulative probability >= top_p. Defaults to `0.95`.
      - `top_k`(`int`, optional): Top-k sampling. Considers the top 'k' most probable tokens. Defaults to `40`.
      - `max_tokens`(`int`, required): The maximum number of new tokens to generate.
      - `min_p`(`float`, optional): Min-p sampling, filters tokens below this probability. Defaults to `0.05`.
      - `stop`(`str | list[str]`, optional): Stop sequences to halt generation. Defaults to `None`.
      - `seed`(`int`, optional): Overrides the model's seed just for this call. Defaults to `None`.
      - `repeat_penalty`(`float`, optional): Penalty for repeating tokens (1.0 = no penalty). Defaults to `1.0`.
      - `presence_penalty`(`float`, optional): Penalty for new tokens (0.0 = no penalty). Defaults to `0.0`.
      - `frequency_penalty`(`float`, optional): Penalty for frequent tokens (0.0 = no penalty). Defaults to `0.0`.
      - `logit_bias`(`dict[int, float]`, optional): Applies a bias to specific tokens. Defaults to `None`.
    - `chat_history_key`(`str`, optional): Key in the packet's generic_data to find
    the conversation history.
    - `rag_context_key`(`str`, optional): Key in the packet's generic_data to find
    RAG context to inject.
    - `system_prompt`(`str | Path`, optional): The system prompt (or path to one)
    to instruct the model.
    - `pattern`(`dict`, optional): A regex pattern used to post-process the model's response.
    - `keep_before`(`bool`, optional): If True, keeps text before the 'pattern' match; otherwise, keeps text after.

    </details>

- **LLaMATextCompletionWithMCP**: Template for text completion with MCP tool integration using LLaMA CPP.

    <details>
    <summary>Attributes</summary>

    - `init_args`(`LLaMAInitArgs`, required): LLaMA model arguments.
      - `llm_model_name`(`str`, required): The name or path of the LLM model to use (e.g. 'TheBloke/Llama-2-7B-GGUF').
      - `llm_model_file`(`str`, required): The specific GGUF model file (e.g., 'llama-2-7b.Q2_K.gguf').
      - `n_gpu_layers`(`int`, optional): Number of layers to offload to the GPU (-1 for all). Defaults to `0`.
      - `use_mmap`(`bool`, optional): Use 'memory-mapping' to load the model. Defaults to `True`.
      - `use_mlock`(`bool`, optional): Force the model to be kept in RAM. Defaults to `False`.
      - `seed`(`int`, optional): RNG seed for model initialization. Defaults to `LLAMA_DEFAULT_SEED`.
      - `n_ctx`(`int`, optional): The context window size. Defaults to `512`.
      - `n_batch`(`int`, optional): The batch size for prompt processing. Defaults to `512`.
      - `n_ubatch`(`int`, optional): The batch size for token generation. Defaults to `512`.
      - `n_threads`(`int`, optional): CPU threads for generation. Defaults to `None`.
      - `n_threads_batch`(`int`, optional): CPU threads for batch processing. Defaults to `None`.
      - `flash_attn`(`bool`, optional): Enable Flash Attention if supported by the GPU. Defaults to `False`.
      - `chat_format`(`str`, optional): Chat template format (e.g., 'chatml'). Defaults to `None`.
      - `verbose`(`bool`, optional): Enable verbose logging from llama.cpp. Defaults to `True`.
    - `completion_args`(`LLaMACompletionArgs`, required): Generation arguments to pass to the selected model.
      - `temperature`(`float`, optional): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
      - `top_p`(`float`, optional): Nucleus sampling. Considers tokens with cumulative probability >= top_p. Defaults to `0.95`.
      - `top_k`(`int`, optional): Top-k sampling. Considers the top 'k' most probable tokens. Defaults to `40`.
      - `max_tokens`(`int`, required): The maximum number of new tokens to generate.
      - `min_p`(`float`, optional): Min-p sampling, filters tokens below this probability. Defaults to `0.05`.
      - `stop`(`str | list[str]`, optional): Stop sequences to halt generation. Defaults to `None`.
      - `seed`(`int`, optional): Overrides the model's seed just for this call. Defaults to `None`.
      - `repeat_penalty`(`float`, optional): Penalty for repeating tokens (1.0 = no penalty). Defaults to `1.0`.
      - `presence_penalty`(`float`, optional): Penalty for new tokens (0.0 = no penalty). Defaults to `0.0`.
      - `frequency_penalty`(`float`, optional): Penalty for frequent tokens (0.0 = no penalty). Defaults to `0.0`.
      - `logit_bias`(`dict[int, float]`, optional): Applies a bias to specific tokens. Defaults to `None`.
    - `chat_history_key`(`str`, optional): Key in the packet's generic_data to find
    the conversation history.
    - `rag_context_key`(`str`, optional): Key in the packet's generic_data to find
    RAG context to inject.
    - `system_prompt`(`str | Path`, optional): The system prompt (or path to one)
    to instruct the model.
    - `pattern`(`dict`, optional): A regex pattern used to post-process the model's response.
    - `keep_before`(`bool`, optional): If True, keeps text before the 'pattern' match; otherwise, keeps text after.
    - `tools_key`(`str`, optional): Key used to extract the raw tools from the data container. Defaults to `""`.
    - `max_tool_retries`(`int`, optional): Maximum consecutive tool execution failures before stopping. Defaults to `3`.
    - `add_tool_to_prompt`(`bool`, optional): Whether to automatically append tool descriptions to the system prompt. Defaults to `True`.

    </details>

- **LLama4TextToText**: Template for text-to-text chat processing using the LLama 4 model.

    <details>
    <summary>Attributes</summary>

    - `init_args`(`LLaMA4InitArgs`, required): LLaMA4 model arguments.
      - `llm_model_name`(`str`, required): The name or path of the LLM model to use (e.g., 'meta-llama/Llama-4-Scout-17B-16E-Instruct').
      - `cache_dir`(`str`, optional): Path to use for the model cache and download.
      - `device_map`(`str`, optional): Device mapping for `from_pretrained`. Defaults to `auto`.
      - `torch_dtype`(`str`, optional): Model tensor precision (e.g., 'auto', 'float16'). Defaults to `auto`.
      - `max_memory`(`dict`, optional): Max memory allocation per device. Defaults to `None`.
    - `completion_args`(`LLMCompletionArgs`, required): Generation arguments to pass to the selected model.
      - `temperature`(`float`, optional): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
      - `top_p`(`float`, optional): Nucleus sampling. Considers tokens with cumulative probability >= top_p. Defaults to `0.95`.
      - `top_k`(`int`, optional): Top-k sampling. Considers the top 'k' most probable tokens. Defaults to `40`.
      - `max_length`(`int`, optional): The maximum length of the sequence (prompt + generation). Defaults to `20`.
      - `max_new_tokens`(`int`, optional): The maximum number of new tokens to generate. Defaults to `None`.
      - `do_sample`(`bool`, optional): Whether to use sampling (True) or greedy decoding (False). Defaults to `True`.
      - `min_p`(`float`, optional): Min-p sampling, filters tokens below this probability. Defaults to `None`.
      - `repetition_penalty`(`float`, optional): Penalty applied to repeated tokens (1.0 = no penalty). Defaults to `1.0`.
    - `chat_history_key`(`str`, optional): Key in the packet's generic_data to find
    the conversation history.
    - `rag_context_key`(`str`, optional): Key in the packet's generic_data to find
    RAG context to inject.
    - `system_prompt`(`str | Path`, optional): The system prompt (or path to one)
    to instruct the model.
    - `pattern`(`dict`, optional): A regex pattern used to post-process the model's response.
    - `keep_before`(`bool`, optional): If True, keeps text before the 'pattern' match; otherwise, keeps text after.

    </details>

- **LLama4MultiModal**: Template for multi modal chat processing using the LLama 4 model.

    <details>
    <summary>Attributes</summary>

    - `init_args`(`LLaMA4InitArgs`, required): LLaMA4 model arguments.
      - `llm_model_name`(`str`, required): The name or path of the LLM model to use (e.g., 'meta-llama/Llama-4-Scout-17B-16E-Instruct').
      - `cache_dir`(`str`, optional): Path to use for the model cache and download.
      - `device_map`(`str`, optional): Device mapping for `from_pretrained`. Defaults to `auto`.
      - `torch_dtype`(`str`, optional): Model tensor precision (e.g., 'auto', 'float16'). Defaults to `auto`.
      - `max_memory`(`dict`, optional): Max memory allocation per device. Defaults to `None`.
    - `completion_args`(`LLMCompletionArgs`, required): Generation arguments to pass to the selected model.
      - `temperature`(`float`, optional): Controls randomness. 0.0 = deterministic, >0.0 = random. Defaults to `0.2`.
      - `top_p`(`float`, optional): Nucleus sampling. Considers tokens with cumulative probability >= top_p. Defaults to `0.95`.
      - `top_k`(`int`, optional): Top-k sampling. Considers the top 'k' most probable tokens. Defaults to `40`.
      - `max_length`(`int`, optional): The maximum length of the sequence (prompt + generation). Defaults to `20`.
      - `max_new_tokens`(`int`, optional): The maximum number of new tokens to generate. Defaults to `None`.
      - `do_sample`(`bool`, optional): Whether to use sampling (True) or greedy decoding (False). Defaults to `True`.
      - `min_p`(`float`, optional): Min-p sampling, filters tokens below this probability. Defaults to `None`.
      - `repetition_penalty`(`float`, optional): Penalty applied to repeated tokens (1.0 = no penalty). Defaults to `1.0`.
    - `chat_history_key`(`str`, optional): Key in the packet's generic_data to find
    the conversation history.
    - `rag_context_key`(`str`, optional): Key in the packet's generic_data to find
    RAG context to inject.
    - `system_prompt`(`str | Path`, optional): The system prompt (or path to one)
    to instruct the model.
    - `pattern`(`dict`, optional): A regex pattern used to post-process the model's response.
    - `keep_before`(`bool`, optional): If True, keeps text before the 'pattern' match; otherwise, keeps text after.

    </details>


> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis Data Tools.

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***LLaMATextCompletion*** use ```sinapsis info --example-template-config LLaMATextCompletion``` to produce the following example config:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: LLaMATextCompletion
  class_name: LLaMATextCompletion
  template_input: InputTemplate
  attributes:
    init_args:
      llm_model_name: '`replace_me:<class ''str''>`'
      llm_model_file: '`replace_me:<class ''str''>`'
      n_gpu_layers: 0
      use_mmap: true
      use_mlock: false
      seed: 4294967295
      n_ctx: 512
      n_batch: 512
      n_ubatch: 512
      n_threads: null
      n_threads_batch: null
      flash_attn: false
      chat_format: null
      verbose: true
    completion_args:
      temperature: 0.2
      top_p: 0.95
      top_k: 40
      max_tokens: '`replace_me:<class ''int''>`'
      min_p: 0.05
      stop: null
      seed: null
      repeat_penalty: 1.0
      presence_penalty: 0.0
      frequency_penalty: 0.0
      logit_bias: null
    chat_history_key: null
    rag_context_key: null
    system_prompt: null
    pattern: null
    keep_before: true
```

<h2 id="example">📚 Usage example</h2>
The following agent passes a text message through a TextPacket and retrieves a response from a LLM
<details id='usage'><summary><strong><span style="font-size: 1.0em;"> Config</span></strong></summary>

```yaml
agent:
  name: chat_completion
  description: Chatbot agent using DeepSeek-R1

templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: TextInput
  class_name: TextInput
  template_input: InputTemplate
  attributes:
    text: what is AI?

- template_name: LLaMATextCompletion
  class_name: LLaMATextCompletion
  template_input: TextInput
  attributes:
    init_args:
      llm_model_name: bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF
      llm_model_file: DeepSeek-R1-Distill-Qwen-7B-Q5_K_S.gguf
      n_ctx: 8192
      n_threads: 8
      n_gpu_layers: -1
      chat_format: chatml
      flash_attn: true
      seed: 10
    completion_args:
      max_tokens: 4096
      temperature: 0.2
      seed: 10
    system_prompt : 'You are a helpful assistant'
    pattern: "</think>"
    keep_before: False
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



