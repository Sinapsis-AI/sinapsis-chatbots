
FROM sinapsis-nvidia:base
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE="1"

RUN apt-get update && apt-get install -y cuda-toolkit-12-4  ninja-build git && apt-get clean

COPY . /app/
WORKDIR /app
RUN uv sync --frozen && uv pip install sinapsis-chatbots[all] --extra-index-url https://pypi.sinapsis.tech
RUN uv pip install flash-attn --no-build-isolation --upgrade