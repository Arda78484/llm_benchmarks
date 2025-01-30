#!/usr/bin/env python3
"""
Benchmark a quantized GGML model using the Ollama LLM API with the OpenAI client.
"""
import os
import time
import argparse
import resource
from openai import OpenAI
import socket
import datetime
from pprint import pp
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Constants
MODELS = ["qwen2.5:0.5b", "qwen2.5:7b", "qwen/qwen2.5-7b-instruct"]
DEFAULT_PROMPTS = [
    "What is the meaning of life?",
    "How many points did you list out?",
    "What is the weather forecast today?",
    "What is the fable involving a fox and grapes?",
    "How to make french fries?",
    "What is the product of 9 and 8?",
    "If a train travels 120 miles in 2 hours, what is its average speed?"
]
DEFAULT_RUNS = 2
DEFAULT_API_CONFIG = {
    "ollama": {
        "name": "Ollama",
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "ollama"
    },
    "mlc": {
        "name": "MLC",
        "api_base": "http://0.0.0.0:9000/v1",
        "api_key": "*"
    },
    "nim": {
        "name": "nim",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key" : "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"
    }
}
CSV_DIR = "../data/"
CSV_HEADER = "timestamp,hostname,api,model,precision,input_tokens,output_tokens,prefill_time,prefill_rate,decode_time,decode_rate,memory\n"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, choices=MODELS)
parser.add_argument('-p', '--prompt', type=str, help="Single prompt override")
parser.add_argument('--runs', type=int, default=DEFAULT_RUNS)
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')
parser.add_argument('--api', type=str, choices=DEFAULT_API_CONFIG, default='ollama', help='Choose the API to use: ollama or mlc')
args = parser.parse_args()
api_config = DEFAULT_API_CONFIG.get(args.api)

def query_model_response(model: str, prompt: str):
    """Sends the prompt to the API and returns response metadata."""

    client = OpenAI(
        base_url = api_config['api_base'],
        api_key = api_config['api_key']
    )

    start_time = time.perf_counter()

    chat = [{
    'role': 'user',
    'content': prompt
    }]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=chat,
            max_tokens=1024,
            timeout=30
        )
    except Exception as e:
        print(f"Error: {e}")
        return None, 0, 0, 0, 0, 0
    
    elapsed_time = time.perf_counter() - start_time

    result = completion.choices[0].message.content
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    
    prefill_rate = prompt_tokens / elapsed_time if elapsed_time > 0 else 0
    decode_rate = completion_tokens / elapsed_time if elapsed_time > 0 else 0

    return result, prompt_tokens, completion_tokens, elapsed_time, prefill_rate, decode_rate

def save_results(file_path, model_name, input_tokens, output_tokens, prefill_rate, decode_rate, memory_usage):
    """Saves benchmarking results to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as file:
            file.write(CSV_HEADER)
    
    with open(file_path, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')},"
                   f"{socket.gethostname()},{api_config['name']},{model_name},"
                   f"fp16,{input_tokens:.1f},{output_tokens:.1f},"
                   f"0,{prefill_rate:},0,{decode_rate:},"
                   f"{memory_usage:}\n")


def benchmark_model(model_name: str, prompts: list, runs: int):
    """Runs benchmarks for a given model and set of prompts."""
    for prompt in prompts:
        print(f"{Fore.CYAN}\n=== Prompt: {prompt} ===\n{Style.RESET_ALL}")
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_elapsed = 0
        total_prefill_rate = 0
        total_decode_rate = 0

        for i in range(runs):
            result, pt, ct, et, pr, dr = query_model_response(model_name, prompt)
            
            if result is None:
                continue

            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_elapsed += et
            total_prefill_rate += pr
            total_decode_rate += dr

            print(f"{Fore.GREEN}[+] Run #{i+1}, Elapsed: {et:.2f}s{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[+] Response:\n{result.strip()}\n{Style.RESET_ALL}")

        # Calculate averages
        avg_prompt = total_prompt_tokens / runs
        avg_completion = total_completion_tokens / runs
        avg_prefill = total_prefill_rate / runs
        avg_decode = total_decode_rate / runs
        total_time = total_elapsed

        print(f"{Fore.MAGENTA}[*] Avg input tokens: {avg_prompt:.1f}")
        print(f"{Fore.MAGENTA}[*] Avg output tokens: {avg_completion:.1f}")
        print(f"{Fore.MAGENTA}[*] Avg prefill rate: {avg_prefill:.2f} tokens/sec")
        print(f"{Fore.MAGENTA}[*] Avg decode rate: {avg_decode:.2f} tokens/sec{Style.RESET_ALL}")

    memory_usage = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    ) / 1024
    print(f"{Fore.RED}Peak memory usage: {memory_usage:.2f} MB{Style.RESET_ALL}")

    if args.save:
        save_results(
            os.path.join(CSV_DIR, args.save),
            model_name,
            avg_prompt,
            avg_completion,
            avg_prefill,
            avg_decode,
            memory_usage
        )

if __name__ == "__main__":
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    if args.model:
        benchmark_model(args.model, prompts, args.runs)
    else:
        for model in MODELS:
            benchmark_model(model, prompts, args.runs)