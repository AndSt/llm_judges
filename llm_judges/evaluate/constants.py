datasets = ["aqua_rat", "gsm8k", "math"]

judge_models = [
    "Qwen/Qwen2-72B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "01-ai/Yi-1.5-34B-Chat-16K",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

models = [
    "Qwen/Qwen2-72B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "01-ai/Yi-1.5-34B-Chat-16K",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-1.1-7b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

model_name_in_paper = {
    "Qwen/Qwen2-72B-Instruct": "Qwen 2 72B",
    "meta-llama/Meta-Llama-3-70B-Instruct": "Llama 3 70B",
    "01-ai/Yi-1.5-34B-Chat-16K": "Yi 1.5 34B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3 8B",
    "google/gemma-1.1-7b-it": "Gemma 1.1 7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B v0.3",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral 7B v0.1",
}

dataset_name_in_paper = {
    "aqua_rat": "AQUA-RAT",
    "gsm8k": "GSM8K",
    "math": "MATH",
}