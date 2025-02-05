# LLM Benchmarking Tool

## Free Memory

To free up memory, run the following command:

```bash
echo "echo 1 > /proc/sys/vm/drop_caches" | sudo sh
```

## Prerequisites

- Docker
- Jetson Containers
- Python

## Installation

1. Install `colorama`:

  ```bash
  pip install colorama
  ```

2. Run the Ollama container:

  ```bash
  jetson-containers run --name ollama $(autotag ollama)
  ```

3. Pull the models:

  ```bash
  docker exec ollama ollama pull qwen2.5:7B
  docker exec ollama ollama pull qwen2.5:0.5B
  ```

## Usage

Ensure that the Ollama container is running and the models are downloaded. Then you can run the benchmarks.

## To Test with MLC

### Running Docker for 0.5B Model

1. Run the Docker container:

  ```bash
  docker run -it --rm \
    --pull=always \
    --gpus all \
    -p 9000:9000 \
    -v ~/.cache:/root/.cache \
    -e HF_HUB_CACHE=/root/.cache/huggingface \
    dustynv/mlc:0.19.2-r36.4.0 \
    sudonim serve \
      --model dusty-nv/Qwen2.5-0.5B-Instruct-q4f16_ft-MLC \
      --quantization q4f16_ft \
      --max-batch-size 1 \
      --host 0.0.0.0 \
      --port 9000
  ```

### Running Docker for 7B Model

1. Run the Docker container:

  ```bash
  docker run -it --rm \
    --pull=always \
    --gpus all \
    -p 9000:9000 \
    -v ~/.cache:/root/.cache \
    -e HF_HUB_CACHE=/root/.cache/huggingface \
    dustynv/mlc:0.19.2-r36.4.0 \
    sudonim serve \
      --model dusty-nv/Qwen2.5-7B-Instruct-q4f16_ft-MLC \
      --quantization q4f16_ft \
      --max-batch-size 1 \
      --max-context-len 2048 \
      --prefill-chunk 1024 \
      --host 0.0.0.0 \
      --port 9000
  ```

2. Run the benchmarks.

## Nim Currently Online

1. Run the benchmarks.