import os
from collections import Counter

import re

import joblib
import numpy as np


def few_shot_result_list_to_per_idx_results(result_list, result_type="float"):
    per_idx_results = {}

    for results in result_list:
        idx = results["idx"]
        if idx not in per_idx_results:
            per_idx_results[idx] = {
                "idx": idx,
                "question": results["question"],
                "answer": results["answer"],
                "truth": results["result"],
                "prompt": results["prompt"],
                "responses": [],
                "result": [],
            }
        try:
            response = results["response"]
            result = response.split("####")
            if result_type == "float":
                # find first occuring (potential floating point number using regex)
                pattern = r"[-+]?\d*\.\d+|\d+"
                result = re.findall(pattern, result[1])
                result = result[0].strip()
                result = float(result)

                if result is not None:
                    per_idx_results[idx]["responses"].append(response)
                    per_idx_results[idx]["result"].append(result)
            elif result_type == "a_to_e":
                value = result[1].strip()
                result = None

                for solution in ["A", "B", "C", "D", "E"]:
                    if solution in value:
                        if result is None:
                            result = solution
                        else:
                            result = False

                if result is not None and result is not False:
                    per_idx_results[idx]["responses"].append(response)
                    per_idx_results[idx]["result"].append(result)

        except Exception as e:
            # print(e)
            # print(results["result"])

            # print("--------------")
            # print(results["response"])
            # print("-------------------------")
            pass

    return per_idx_results


def majority_vote(results):
    if len(results) == 0:
        return None
    # return max(set(results), key=results.count)
    counter = Counter(results)
    max_occurences = max(counter.values())
    max_occurence_results = [k for k, v in counter.items() if v == max_occurences]
    if len(max_occurence_results) == 1:
        return max_occurence_results[0]
    else:
        # random sample
        return float(np.random.choice(max_occurence_results))


def get_avg_numbers(per_idx_results):
    avg_code_results = np.mean(
        [len(results["result"]) for results in per_idx_results.values()]
    )
    print(f"Average number of code results: {avg_code_results}")

    code_exists_acc = []
    code_avg_acc = []
    code_avg_mv_acc = []

    for i, results in per_idx_results.items():
        truth = results["truth"]

        code_result = results["result"]

        sample_accs = []
        for res in code_result:
            sample_accs.append(truth == res)

        if len(sample_accs) > 0:
            code_avg_acc.append(np.mean(sample_accs))
        else:
            code_avg_acc.append(0)

        mv = majority_vote(code_result)
        if truth == mv:
            code_avg_mv_acc.append(1)
        else:
            # print(i, truth, mv, code_result)
            code_avg_mv_acc.append(0)

        if code_result is not None:
            code_exists_acc.append(truth in code_result)
        else:
            code_exists_acc.append(0)

    print(f"Average code exists acc: {np.mean(code_exists_acc)}")
    print(f"Average code acc: {np.mean(code_avg_acc)}")
    print(f"Average code mv acc: {np.mean(code_avg_mv_acc)}")


def load_generations_per_idx(initial_dir, dataset, model):
    model_load_name = model.replace("/", "_")
    load_file = os.path.join(initial_dir, dataset, model_load_name, "initial.pbz2")
    generations = joblib.load(load_file)
    if dataset == "aqua_rat":
        per_idx_results = few_shot_result_list_to_per_idx_results(generations, "a_to_e")
    else:
        per_idx_results = few_shot_result_list_to_per_idx_results(generations)

    return per_idx_results
