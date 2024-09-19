from typing import List
import os

from tqdm import tqdm

import joblib
import json

import re
import string


def remove_whitespace(my_string):
    remove_chars = string.whitespace  # Includes spaces, newlines, tabs, etc.
    translation_table = {ord(char): None for char in remove_chars}
    new_string = my_string.translate(translation_table)
    return new_string


def parse_json_comparison_response(response):
    response = remove_whitespace(response)
    try:
        json_pattern = r"```json.*```"
        json_objects = re.findall(json_pattern, response, re.DOTALL)

        obj = (
            json_objects[0].replace("```json", "").replace("```", "").replace("\_", "_")
        )
        obj = json.loads(obj)
        correct_answer = obj.get("answer", "C")
        if "A" in correct_answer and "B" not in correct_answer:
            correct_answer = "A"
        elif "B" in correct_answer and "A" not in correct_answer:
            correct_answer = "B"
        if correct_answer not in ["A", "B"]:
            raise Exception("Invalid correct answer")
        return correct_answer
    except:
        return None


def parse_correctness(generations):
    num_errs = 0

    for i in range(len(generations)):
        gen = generations[i]
        comp = gen["comparison"]
        response = parse_json_comparison_response(gen["generation"]["response"])
        generations[i]["extracted"] = response
        if response not in ["A", "B"]:
            generations[i]["correct_parse"] = False
            generations[i]["correct"] = False
            num_errs += 1
            # print(f"Error at generation {i}", gen["generation"]["response"])
            continue
        generations[i]["correct_parse"] = True
        if comp["meta"] == "a_b":
            if response == "A":
                if comp["model_1_opinion"] == comp["truth"]:
                    generations[i]["correct"] = True
                else:
                    generations[i]["correct"] = False
            else:
                if comp["model_2_opinion"] == comp["truth"]:
                    generations[i]["correct"] = True
                else:
                    generations[i]["correct"] = False
        elif comp["meta"] == "b_a":
            if response == "B":
                if comp["model_1_opinion"] == comp["truth"]:
                    generations[i]["correct"] = True
                else:
                    generations[i]["correct"] = False
            else:
                if comp["model_2_opinion"] == comp["truth"]:
                    generations[i]["correct"] = True
                else:
                    generations[i]["correct"] = False
        else:
            generations[i]["correct"] = False
            num_errs += 1

    return generations, num_errs


def load_comparisons_in_dict(
    judge_models: List[str],
    datasets: List[str],
    data_dir: str,
    exclude_wrong_parses: bool = True,
):
    num_files = 0
    for judge in judge_models:
        judge_file_name = judge.replace("/", "_")
        for dataset in datasets:
            comparison_dir = os.path.join(data_dir, dataset, judge_file_name)
            for file in os.listdir(comparison_dir):
                if not file.endswith(".pbz2"):
                    continue
                num_files += 1

    # corr full
    # P_D(J=T | A, B)
    correct_full = {}
    correct_a_neq_b = {}

    # corr partial
    # P_D(J=T | A=X, B=Y)
    correct_partial = {}

    # counts = {}
    counts = {}
    unnormalized_counts = {}
    unparsed = {}

    all_generations = {}

    pbar = tqdm(total=num_files)

    for judge in judge_models:
        correct_full[judge] = {}
        correct_a_neq_b[judge] = {}
        correct_partial[judge] = {}
        counts[judge] = {}
        unnormalized_counts[judge] = {}
        unparsed[judge] = {}

        all_generations[judge] = {}

        judge_file_name = judge.replace("/", "_")

        for dataset in datasets:

            correct_full[judge][dataset] = {}
            correct_a_neq_b[judge][dataset] = {}
            correct_partial[judge][dataset] = {}
            counts[judge][dataset] = {}
            unnormalized_counts[judge][dataset] = {}
            unparsed[judge][dataset] = {}
            all_generations[judge][dataset] = {}

            comparison_dir = os.path.join(data_dir, dataset, judge_file_name)

            for file in os.listdir(comparison_dir):
                if not file.endswith(".pbz2"):
                    continue
                comparisons = joblib.load(os.path.join(comparison_dir, file))[
                    "generations"
                ]
                comparisons, num_errs = parse_correctness(comparisons)

                pair = (
                    comparisons[0]["comparison"]["model_1"],
                    comparisons[0]["comparison"]["model_2"],
                )

                correct_full[judge][dataset][pair] = {}
                correct_a_neq_b[judge][dataset][pair] = {}
                correct_partial[judge][dataset][pair] = {}
                counts[judge][dataset][pair] = {}
                unnormalized_counts[judge][dataset][pair] = {}
                unparsed[judge][dataset][pair] = {}

                splitted_generations = {
                    (True, True): [],
                    (True, False): [],
                    (False, True): [],
                    (False, False): [],
                }

                for comparison in comparisons:
                    if exclude_wrong_parses:
                        if not comparison["correct_parse"]:
                            continue
                    m1_corr = (
                        comparison["comparison"]["model_1_opinion"]
                        == comparison["comparison"]["truth"]
                    )
                    m2_corr = (
                        comparison["comparison"]["model_2_opinion"]
                        == comparison["comparison"]["truth"]
                    )
                    splitted_generations[(m1_corr, m2_corr)].append(comparison)

                all_generations[judge][dataset][pair] = splitted_generations

                # correct full
                preds = []
                for key, value in splitted_generations.items():
                    preds.extend(
                        [
                            gen["correct"] if gen["correct_parse"] else False
                            for gen in value
                        ]
                    )
                correct_full[judge][dataset][pair] = sum(preds) / len(preds)

                preds = []
                for key, value in splitted_generations.items():
                    preds.extend(
                        [
                            gen["correct"] if gen["correct_parse"] else False
                            for gen in value
                            if gen["comparison"]["model_1_opinion"]
                            != gen["comparison"]["model_2_opinion"]
                        ]
                    )
                correct_a_neq_b[judge][dataset][pair] = sum(preds) / len(preds)

                # correct partial
                for key, value in splitted_generations.items():
                    preds = [
                        gen["correct"] if gen["correct_parse"] else False
                        for gen in value
                    ]
                    if len(preds) == 0:
                        correct_partial[judge][dataset][pair][key] = 0
                    else:
                        correct_partial[judge][dataset][pair][key] = sum(preds) / len(
                            preds
                        )

                # counts
                full_parsed = sum([gen["correct_parse"] for gen in comparisons])
                for key, value in splitted_generations.items():
                    counts[judge][dataset][pair][key] = len(value) / full_parsed
                    unnormalized_counts[judge][dataset][pair][key] = len(value)

                for key, value in splitted_generations.items():
                    unparsed[judge][dataset][pair][key] = len(
                        [gen for gen in value if not gen["correct_parse"]]
                    )

                pbar.update(1)

    return (
        correct_full,
        correct_a_neq_b,
        correct_partial,
        counts,
        unnormalized_counts,
        unparsed,
        all_generations,
    )
