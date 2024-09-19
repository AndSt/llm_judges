import argparse
from typing import List, Tuple, Dict
import os

import uuid
import time
import tqdm

import math
import joblib

# istarmap.py for Python 3.8+
import multiprocessing
import multiprocessing.pool as mpp

from llm_judges.evaluate.initial import load_generations_per_idx
from llm_judges.judgements.prompt import judgement_inference_wrapper

from llm_judges.connection import istarmap

mpp.Pool.istarmap = istarmap


RANDOM_STATE = 42


def get_comparison_of_idx(
    per_idx_results_model_1: Dict,
    per_idx_results_model_2: Dict,
    idx: int,
    model_1_name: str,
    model_2_name: str,
    judge_name: str,
    pairs: List[Tuple[str, str]],
):

    comparisons = []

    for pair in pairs:
        uid = str(uuid.uuid4())

        comparison = {
            "model_1": model_1_name,
            "model_2": model_2_name,
            "judge": judge_name,
            "meta": f"a_b",
            "model_1_opinion": per_idx_results_model_1[idx]["result"][pair[0]],
            "model_2_opinion": per_idx_results_model_2[idx]["result"][pair[1]],
            "model_1_id": pair[0],
            "model_2_id": pair[1],
            "idx": idx,
            "uuid": uid,
        }
        comparisons.append(comparison)

        # reverse
        comparison = {
            "model_1": model_1_name,
            "model_2": model_2_name,
            "judge": judge_name,
            "meta": f"b_a",
            "model_1_opinion": per_idx_results_model_1[idx]["result"][pair[0]],
            "model_2_opinion": per_idx_results_model_2[idx]["result"][pair[1]],
            "model_1_id": pair[0],
            "model_2_id": pair[1],
            "idx": idx,
            "uuid": uid,
        }
        comparisons.append(comparison)

    return comparisons


def generate_args_list(
    all_comparisons,
    comparison_type: str = "llama3",
    url="http://localhost:8080/v1",
    num_responses_per_sample=5,
    temperature=0.9,
):

    args_list = []

    for comparison in all_comparisons:
        single_arg = [comparison_type, comparison, comparison["question"]]
        if comparison["meta"] == "a_b":
            single_arg.append(comparison["model_1_opinion_str"])
            single_arg.append(comparison["model_2_opinion_str"])
        elif comparison["meta"] == "b_a":
            single_arg.append(comparison["model_2_opinion_str"])
            single_arg.append(comparison["model_1_opinion_str"])

        single_arg.extend(
            [
                url,
                comparison["judge"],
                temperature,
            ]
        )
        args_list.append(single_arg)

    args_list = args_list * num_responses_per_sample

    return args_list


def main(
    original_data_path: str,
    config_file_path: str,
    save_path: str,
    model_name: str,
    url: str = "http://localhost:8080/v1",
    comparison_type: str = "std_comparison",
    num_processes: int = 200,
    num_responses_per_sample: int = 11,
    temperature: float = 0.9,
):

    config_dict = joblib.load(config_file_path)

    j = 0
    for (model_1_name, model_2_name), dataset_dict in config_dict["full_dict"].items():
        j += 1
        print(
            f"Running {j}/{len(config_dict['full_dict'])}: {model_1_name} vs {model_2_name}"
        )
        for dataset, idx_dict in dataset_dict.items():

            if dataset == "predictions_per_pair":
                continue

            model_1_save_name = model_1_name.replace("/", "_")
            model_2_save_name = model_2_name.replace("/", "_")
            save_dir = os.path.join(save_path, dataset, model_name.replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)

            full_save_file = os.path.join(
                save_dir,
                f"{model_1_save_name}_vs_{model_2_save_name}_comparison.pbz2",
            )
            if os.path.exists(full_save_file):
                print("File exists, skipping")
                continue

            per_idx_generations = {
                model_1_name: load_generations_per_idx(
                    initial_dir=original_data_path,
                    dataset=dataset,
                    model=model_1_name,
                ),
                model_2_name: load_generations_per_idx(
                    initial_dir=original_data_path,
                    dataset=dataset,
                    model=model_2_name,
                ),
            }

            comparisons = []

            for idx, idx_data in idx_dict.items():

                idx_comparisons = get_comparison_of_idx(
                    per_idx_results_model_1=per_idx_generations[model_1_name],
                    per_idx_results_model_2=per_idx_generations[model_2_name],
                    idx=idx,
                    model_1_name=model_1_name,
                    model_2_name=model_2_name,
                    judge_name=model_name,
                    pairs=idx_data["sampled_tuples"],
                )

                if idx_comparisons is not None:
                    for i in range(len(idx_comparisons)):

                        idx_comparisons[i]["idx"] = idx
                        idx_comparisons[i]["dataset"] = dataset
                        idx_comparisons[i]["question"] = per_idx_generations[
                            idx_comparisons[i]["model_1"]
                        ][idx]["question"]
                        idx_comparisons[i]["answer"] = per_idx_generations[
                            idx_comparisons[i]["model_1"]
                        ][idx]["answer"]
                        idx_comparisons[i]["truth"] = per_idx_generations[
                            idx_comparisons[i]["model_1"]
                        ][idx]["truth"]

                        model_1_opinion = idx_comparisons[i]["model_1_opinion"]
                        model_2_opinion = idx_comparisons[i]["model_2_opinion"]

                        if isinstance(model_1_opinion, float) and math.isfinite(
                            model_1_opinion
                        ):
                            if model_1_opinion == int(model_1_opinion):
                                model_1_opinion = int(model_1_opinion)
                        model_1_opinion_st = str(model_1_opinion)
                        if isinstance(model_2_opinion, float) and math.isfinite(
                            model_2_opinion
                        ):
                            if model_2_opinion == int(model_2_opinion):
                                model_2_opinion = int(model_2_opinion)
                        model_2_opinion_st = str(model_2_opinion)

                        model_1_opinion_str = per_idx_generations[
                            idx_comparisons[i]["model_1"]
                        ][idx]["responses"][idx_comparisons[i]["model_1_id"]]
                        model_2_opinion_str = per_idx_generations[
                            idx_comparisons[i]["model_2"]
                        ][idx]["responses"][idx_comparisons[i]["model_2_id"]]

                        model_1_opinion_str2 = model_1_opinion_str.replace(
                            f" {model_1_opinion_st}", f" {model_2_opinion_st}"
                        )
                        model_2_opinion_str2 = model_2_opinion_str.replace(
                            f" {model_2_opinion_st}", f" {model_1_opinion_st}"
                        )

                        model_1_opinion_str2 = model_1_opinion_str2.replace(
                            f"{{{model_1_opinion_st}", f"{{{model_2_opinion_st}"
                        )
                        model_2_opinion_str2 = model_2_opinion_str2.replace(
                            f"{{{model_2_opinion_st}", f"{{{model_1_opinion_st}"
                        )

                        model_1_opinion_str2 = model_1_opinion_str2.replace(
                            f"={model_1_opinion_st}", f"={model_2_opinion_st}"
                        )
                        model_2_opinion_str2 = model_2_opinion_str2.replace(
                            f"={model_2_opinion_st}", f"={model_1_opinion_st}"
                        )

                        idx_comparisons[i]["model_1_opinion_str"] = model_1_opinion_str2
                        idx_comparisons[i]["model_2_opinion_str"] = model_2_opinion_str2

                    comparisons.extend(idx_comparisons)

            args_list_full = generate_args_list(
                all_comparisons=comparisons,
                comparison_type=comparison_type,
                url=url,
                num_responses_per_sample=num_responses_per_sample,
                temperature=temperature,
            )

            print(f"Length of args list: {len(args_list_full)}")

            start_time = time.time()

            generations = []
            with multiprocessing.Pool(processes=num_processes) as pool:

                for res in tqdm.tqdm(
                    pool.istarmap(judgement_inference_wrapper, args_list_full),
                    total=len(args_list_full),
                ):
                    generations.append(res)

            joblib.dump(
                {
                    "generations": generations,
                    "original_data_path": original_data_path,
                    "judge": model_name,
                    "model_1": model_1_name,
                    "model_2": model_2_name,
                    "generation_type": comparison_type,
                    "num_responses_per_sample": num_responses_per_sample,
                    "temperature": temperature,
                    "num_processes": num_processes,
                    "runtime": time.time() - start_time,
                },
                full_save_file,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_data_path",
        type=str,
        help="Path to the original data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/dtw_admin/andi/projects/agent_fun/notebooks/data",
        help="Path to save the data",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="/home/dtw_admin/andi/projects/agent_fun/notebooks/data/comparison_config.pbz2",
        help="Path to the config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/v1",
        help="URL for the model",
    )
    parser.add_argument(
        "--comparison_type",
        type=str,
        default="std_comparison",
        help="Comparison type",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=100,
        help="Number of processes",
    )
    parser.add_argument(
        "--num_responses_per_sample",
        type=int,
        default=10,
        help="Number of responses per sample",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    args = parser.parse_args()

    main(
        original_data_path=args.original_data_path,
        save_path=args.save_path,
        config_file_path=args.config_file_path,
        model_name=args.model_name,
        url=args.url,
        comparison_type=args.comparison_type,
        num_processes=args.num_processes,
        num_responses_per_sample=args.num_responses_per_sample,
        temperature=args.temperature,
    )
