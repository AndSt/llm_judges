import os
import argparse

import tqdm
import time

import joblib

import multiprocessing
import multiprocessing.pool as mpp

from llm_judges.candidate_answers.few_shot_prompt import (
    generate_few_shot_answer,
    generate_few_shot_answer_llama3,
)
from llm_judges.datasets import gsm8k, math, aqua_rat

from llm_judges.connection import istarmap

mpp.Pool.istarmap = istarmap


def get_few_shot_samples(dset, get_question_answer_result_fct, num_samples=5):

    few_shots = []

    for idx in range(0, min(num_samples, len(dset))):
        question, answer, _ = get_question_answer_result_fct(dset[idx])

        few_shots.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    return few_shots


def main(
    data_path: str,
    model_name: str,
    dataset_name: str,
    url: str = "http://localhost:8080/v1",
    num_processes: int = 200,
    num_responses_per_sample: int = 10,
    temperature: float = 0.9,
    debug: bool = False,
):

    # loads only a subset if debugging is on
    if dataset_name == "gsm8k":
        dset, few_shot_dset = gsm8k.load_initial_data(num_few_shots=5, debug=debug)
        get_question_answer_result_fct = gsm8k.get_question_answer_result
    elif dataset_name == "math":
        dset, few_shot_dset = math.load_initial_data(num_few_shots=5, debug=debug)
        get_question_answer_result_fct = math.get_question_answer_result
    # elif dataset_name == "mmlu_high_school_math":
    #     dset, few_shot_dset = mmlu_high_school_math.load_initial_data(
    #         num_few_shots=5, debug=debug
    #     )
    #     get_question_answer_result_fct = mmlu_high_school_math.get_question_answer_result
    elif dataset_name == "aqua_rat":
        dset, few_shot_dset = aqua_rat.load_initial_data(num_few_shots=5, debug=debug)
        get_question_answer_result_fct = aqua_rat.get_question_answer_result
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Prepare data for inference
    few_shots = get_few_shot_samples(few_shot_dset, get_question_answer_result_fct)

    args = []

    for idx in range(0, len(dset)):

        question, answer, result = get_question_answer_result_fct(dset[idx])
        tuple_ = (
            idx,
            question,
            answer,
            result,
            model_name,
            temperature,
            url,
            few_shots,
        )
        args.extend([tuple_] * num_responses_per_sample)

    # prepare generation
    save_dir = os.path.join(
        data_path, "initial_few_shot", dataset_name, model_name.replace("/", "_")
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")

    # define inference function
    if model_name in [
        "01-ai/Yi-1.5-34B-Chat-16K",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "google/gemma-1.1-7b-it",
        "Qwen/Qwen2-72B-Instruct",
    ]:
        generate_fct = generate_few_shot_answer
    elif model_name in [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]:
        generate_fct = generate_few_shot_answer_llama3
    else:
        raise ValueError(f"Model {model_name} not found")

    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []

        for res in tqdm.tqdm(pool.istarmap(generate_fct, args), total=len(args)):
            results.append(res)

    joblib.dump(
        {
            "results": results,
            "model_name": model_name,
            "url": url,
            "generation_type": "few_shot",
            "num_responses_per_sample": num_responses_per_sample,
            "num_processes": num_processes,
            "temperature": temperature,
            "few_shot_samples": few_shots,
            "runtime": time.time() - start_time,
        },
        os.path.join(save_dir, f"initial.pbz2"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add data path
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/stephana93dm/main/projects/agent_fun/data",
        help="Data path",
    )
    # add url
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/v1",
        help="URL for API",
    )
    # add model
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model to use",
    )
    # add dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Dataset to use",
    )
    parser.add_argument(
        "--num_responses_per_sample", type=int, default=10, help="Number of samples"
    )
    # add num processes
    parser.add_argument(
        "--num_processes",
        type=int,
        default=200,
        help="Number of processes",
    )
    # add temperature
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature for generation",
    )
    # add debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        url=args.url,
        num_processes=args.num_processes,
        num_responses_per_sample=args.num_responses_per_sample,
        temperature=args.temperature,
        debug=args.debug,
    )
