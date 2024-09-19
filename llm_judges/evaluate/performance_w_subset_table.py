# import os
# import json
# import joblib

# from numpy import save
# import pandas as pd
# from tqdm import tqdm

# from agent_fun.code_comparison.evaluate import get_avg_numbers
# from agent_fun.comparison.transform_comparison import (
#     few_shot_result_list_to_per_idx_results,
# )


# def load_generations_per_idx(initial_dir, dataset, model):
#     model_load_name = model.replace("/", "_")
#     load_file = os.path.join(initial_dir, dataset, model_load_name, "initial.pbz2")
#     generations = joblib.load(load_file)["results"]
#     if dataset == "aqua_rat":
#         per_idx_results = few_shot_result_list_to_per_idx_results(generations, "a_to_e")
#     else:
#         per_idx_results = few_shot_result_list_to_per_idx_results(generations)

#     return per_idx_results


# def create_performance_w_judge_idx_subset_table(
#     models, datasets, initial_dir, save_dir, compared_ids
# ):

#     progress = tqdm(total=len(datasets) * len(models))

#     rows = []

#     for model in models:
#         for dataset in datasets:
#             print(dataset, model)
#             per_idx_results = load_generations_per_idx(initial_dir, dataset, model)

#             results = get_avg_numbers(per_idx_results)

#             row = {
#                 "model": model,
#                 "dataset": dataset,
#                 "num_samples": len(per_idx_results),
#                 "type": "full",
#                 "answer exists": round(100 * results["avg_exists_acc"], 2),
#                 "avg performance": round(100 * results["avg_acc"], 2),
#                 "mv performance": round(100 * results["avg_mv_acc"], 2),
#             }
#             rows.append(row)

#             per_idx_subset = {
#                 k: v for k, v in per_idx_results.items() if k in compared_ids[dataset]
#             }

#             results = get_avg_numbers(per_idx_subset)

#             row = {
#                 "model": model,
#                 "dataset": dataset,
#                 "num_samples": len(per_idx_subset),
#                 "type": "subset",
#                 "answer exists": round(100 * results["avg_exists_acc"], 2),
#                 "avg performance": round(100 * results["avg_acc"], 2),
#                 "mv performance": round(100 * results["avg_mv_acc"], 2),
#             }
#             rows.append(row)

#             progress.update(1)

#     df = pd.DataFrame(rows)

#     df.to_csv(
#         os.path.join(
#             save_dir,
#             f"/avg_performance_{judge_model_name}_ids.csv",
#             index=False,
#             sep=";",
#         )
#     )


# if __name__ == "__main__":

#     # create_performance_w_judge_idx_subset_table(
#     #     models, datasets, initial_dir, compared_ids
#     # )

#     datasets = ["aqua_rat", "gsm8k", "math"]
#     models = [
#         "01-ai/Yi-1.5-34B-Chat-16K",
#         "mistralai/Mistral-7B-Instruct-v0.1",
#         "mistralai/Mistral-7B-Instruct-v0.3",
#         "mistralai/Mixtral-8x7B-Instruct-v0.1",
#         "google/gemma-1.1-7b-it",
#         "Qwen/Qwen2-72B-Instruct",
#         "meta-llama/Meta-Llama-3-8B-Instruct",
#         "meta-llama/Meta-Llama-3-70B-Instruct",
#     ]

#     judge_models = [
#         "01-ai/Yi-1.5-34B-Chat-16K",
#         "mistralai/Mistral-7B-Instruct-v0.1",
#         "mistralai/Mistral-7B-Instruct-v0.3",
#         "mistralai/Mixtral-8x7B-Instruct-v0.1",
#         "google/gemma-1.1-7b-it",
#         "Qwen/Qwen2-72B-Instruct",
#         "meta-llama/Meta-Llama-3-8B-Instruct",
#         "meta-llama/Meta-Llama-3-70B-Instruct",
#     ]

#     initial_dir = "/data/stephana93dm/main/projects/agent_fun/data/initial_few_shot_plain"
#     save_dir = "/data/stephana93dm/main/projects/agent_fun/notebooks/tables"

#     for judge_model in judge_models:
        
#         judge_model_name = judge_model.replace("/", "_")

#         with open(
#             f"/data/stephana93dm/main/projects/agent_fun/notebooks/tables/{judge_model_name}_compared_ids_per_dset.json"
#         ) as f:
#             compared_ids = json.load(f)

#         create_performance_w_judge_idx_subset_table(
#             models, datasets, initial_dir, save_dir, compared_ids
#         )
