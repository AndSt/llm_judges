from collections import Counter

import json
import numpy as np
import random
import string
import uuid

import re


def opinions_to_counts_ids(opinions):
    if isinstance(opinions, list):
        opinions = {i: o for i, o in enumerate(opinions)}
    opinion_list = list(opinions.values())
    counts = Counter(opinion_list).most_common()
    count_dict = [
        {
            "opinion": c[0],
            "count": c[1],
            "ids": [k for k, v in opinions.items() if v == c[0]],
        }
        for c in counts
    ]
    return count_dict


def get_corr_incorr(counts, idx_info):
    truth = idx_info["truth"]

    count_dict = {c["opinion"]: c for c in counts}
    if truth not in count_dict or len(count_dict) < 2:
        return None

    # add correct vs incorrect
    corr_id = random.sample(count_dict[truth]["ids"], 1)[0]
    incorr_opinion = random.sample(
        [c["opinion"] for c in counts if c["opinion"] != truth], 1
    )[0]
    incorr_idx = random.sample(count_dict[incorr_opinion]["ids"], 1)[0]

    return {
        "corr_opinion": truth,
        "incorr_opinion": incorr_opinion,
        "corr_id": corr_id,
        "incorr_id": incorr_idx,
    }


def create_a_b_comparisons(counts, idx_info):

    truth = idx_info["truth"]

    count_dict = {c["opinion"]: c for c in counts}
    comparisons = []
    if truth not in count_dict:
        return comparisons
    if len(count_dict) < 3:
        return comparisons

    # add correct vs incorrect
    corr_id = random.sample(count_dict[truth]["ids"], 1)[0]
    incorr_opinion = random.sample(
        [c["opinion"] for c in counts if c["opinion"] != truth], 1
    )[0]
    incorr_idx = random.sample(count_dict[incorr_opinion]["ids"], 1)[0]

    uid = str(uuid.uuid4())
    comparisons.append(
        {
            "opinion_1": truth,
            "opinion_2": incorr_opinion,
            "id_1": corr_id,
            "id_2": incorr_idx,
            "meta": "a_b_cor_incorr",
            "comp_id": uid,
        }
    )
    comparisons.append(
        {
            "opinion_1": incorr_opinion,
            "opinion_2": truth,
            "id_1": incorr_idx,
            "id_2": corr_id,
            "meta": "a_b_incorr_corr",
            "comp_id": uid,
        }
    )

    return comparisons


def create_gt_a_truth_comparisons(counts, idx_info):
    truth = idx_info["truth"]

    incorr_count_dict = {c["opinion"]: c for c in counts if c["opinion"] != truth}
    comparisons = []

    if len(incorr_count_dict) < 1:
        return comparisons

    # add correct vs incorrect
    incorr_opinion = random.sample(
        [c["opinion"] for c in counts if c["opinion"] != truth], 1
    )[0]
    incorr_idx = random.sample(incorr_count_dict[incorr_opinion]["ids"], 1)[0]

    uid = str(uuid.uuid4())
    comparisons.append(
        {
            "opinion_1": truth,
            "opinion_2": incorr_opinion,
            "id_1": -1,
            "id_2": incorr_idx,
            "meta": "a_b_cor_incorr",
            "comp_id": uid,
        }
    )
    comparisons.append(
        {
            "opinion_1": incorr_opinion,
            "opinion_2": truth,
            "id_1": incorr_idx,
            "id_2": -1,
            "meta": "a_b_incorr_corr",
            "comp_id": uid,
        }
    )

    return comparisons


def create_comparisons(counts, *args, **kwargs):
    comparisons = []
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            i_ids = random.sample(counts[i]["ids"], min(2, len(counts[i]["ids"])))
            j_ids = random.sample(counts[j]["ids"], min(2, len(counts[j]["ids"])))
            for ii in i_ids:
                for jj in j_ids:
                    for _ in range(2):
                        comparisons.append(
                            {
                                "opinion_1": counts[i]["opinion"],
                                "opinion_2": counts[j]["opinion"],
                                "id_1": ii,
                                "id_2": jj,
                            }
                        )
                        comparisons.append(
                            {
                                "opinion_1": counts[j]["opinion"],
                                "opinion_2": counts[i]["opinion"],
                                "id_1": jj,
                                "id_2": ii,
                            }
                        )
    return comparisons


def get_idx_comparison_data(
    idx, per_idx_results, url, model_name, temperature, counts_to_comparisons_fct
):
    count_ids = opinions_to_counts_ids(per_idx_results[idx]["result"])
    comparisons = counts_to_comparisons_fct(count_ids, per_idx_results[idx])
    if len(comparisons) == 0:
        print("No comparisons for idx: ", idx)
        print(per_idx_results[idx]["result"])

    for i in range(0, len(comparisons)):
        comparisons[i]["sample_id"] = idx
        comparisons[i]["question"] = per_idx_results[idx]["question"]
        comparisons[i]["answer"] = per_idx_results[idx]["answer"]
        comparisons[i]["truth"] = per_idx_results[idx]["truth"]

    truth_str = (
        str(int(per_idx_results[idx]["truth"]))
        if int(per_idx_results[idx]["truth"]) == per_idx_results[idx]["truth"]
        else str(per_idx_results[idx]["truth"])
    )
    args_list = [
        [
            comparison,
            per_idx_results[idx]["question"],
            (
                per_idx_results[idx]["responses"][comparison["id_1"]]
                if comparison["id_1"] != -1
                else per_idx_results[idx]["answer"] + "#### " + truth_str
            ),
            (
                per_idx_results[idx]["responses"][comparison["id_2"]]
                if comparison["id_2"] != -1
                else per_idx_results[idx]["answer"] + "#### " + truth_str
            ),
            url,
            model_name,
            temperature,
        ]
        for comparison in comparisons
    ]
    return args_list



# def transform_generation(text):
#     final_answer = None
#     try:

#         # Define your JSON detection regex pattern
#         json_pattern = r"\{.*\}"
#         json_objects = re.findall(json_pattern, text, re.DOTALL)
#         text = json_objects[0]

#         text = text.replace("\\", "")
#         # Remove trailing commas within arrays
#         text = re.sub(r",\s*]", "]", text)

#         # Remove trailing commas within objects
#         text = re.sub(r",\s*}", "}", text)
#         answer = json.loads(text, strict=False)

#         # check if answer is valid
#         if "answer" in answer:
#             final_answer = answer["answer"]
#         elif "step_5" in answer:
#             if "answer" in answer["step_5"]:
#                 final_answer = answer["step_5"]["answer"]
#             elif "final result" in answer:
#                 final_answer = answer["final result"]
#             elif "final result" in answer["step_5"]:
#                 final_answer = answer["step_5"]["final result"]
#         elif "synthesis" in answer:
#             if "answer" in answer["synthesis"]:
#                 final_answer = answer["synthesis"]["answer"]
#         else:
#             print("wuuh")
#             print("----------------")
#             print(answer)

#         if isinstance(final_answer, str):
#             final_answer = final_answer.replace("$", "")

#         if final_answer is not None:
#             final_answer = float(final_answer)
#         else:
#             final_answer = None
#     except Exception as e:
#         final_answer = None
#         print(e)

#     return final_answer


# def transform_generations(generations):
#     answers = []

#     for gen in generations:
#         final_answer = None
#         for i in [0]:
#             if final_answer is not None:
#                 continue
#             answer = gen["generation"].choices[i].message.content
#             final_answer = transform_generation(answer)
#             if final_answer is not None:
#                 answers.append(
#                     (
#                         gen["comparison"]["opinion_1"],
#                         gen["comparison"]["opinion_2"],
#                         final_answer,
#                     )
#                 )
#     # print(answers)
#     # print(len(answers))
#     # print(Counter([a[0] for a in answers]).most_common())
#     # print(Counter([a[2] for a in answers]).most_common())

#     return answers


def make_triplet_comparisons(answers):
    # Assuming you have a list of triplets
    # Absorb the 3rd value into a list
    new_triplets = {}
    for a, b, c in answers:
        a = float(a)
        b = float(b)
        c = float(c)

        if (a, b) in new_triplets:
            new_triplets[(a, b)].append(c)
        elif (b, a) in new_triplets:
            new_triplets[(b, a)].append(c)
        else:
            new_triplets[(a, b)] = [c]

    # transform to probabilities
    for k, v in new_triplets.items():
        new_triplets[k] = {a: b for (a, b) in Counter(v).most_common()}
    # print("-----------------")
    # for k, v in new_triplets.items():
    #     print(f"{k}: {v}")

    # print("-----------------")
    triplet_comparisons = []
    for k, v in new_triplets.items():
        v = v.copy()
        # v = [elt for elt in v if elt[0] in k]
        if k[0] in v:
            if k[1] in v:
                if v[k[0]] <= v[k[1]]:
                    triplet_comparisons.append((k[0], k[1]))
                if v[k[0]] >= v[k[1]]:
                    triplet_comparisons.append((k[1], k[0]))
            else:
                triplet_comparisons.append((k[1], k[0]))
        elif k[1] in v:
            triplet_comparisons.append((k[0], k[1]))
        # if len(v) == 0:
        #     continue
        # elif len(v) == 1:
        #     if k[0] == k[1]:
        #         continue
        #     if v[0][0] == k[0]:
        #         triplet_comparisons.append((k[1], k[0]))
        #     elif v[0][0] == k[1]:
        #         triplet_comparisons.append((k[0], k[1]))
        # else:
        #     if v[0][1] > v[1][1]:
        #         if v[0][0] == k[0]:
        #             triplet_comparisons.append((k[1], k[0]))
        #         elif v[0][0] == k[1]:
        #             triplet_comparisons.append((k[0], k[1]))

        # if v[0][1] == v[1][1]:
        #     triplet_comparisons.append((k[1], k[0]))
        #     triplet_comparisons.append((k[0], k[1]))

    most_winns = {}
    # for k, v in new_triplets.items():
    #     for a, b in v:
    #         if a in most_winns:
    #             most_winns[a] += b / sum([z[1] for z in v])
    #         else:
    #             most_winns[a] = b
    # print("-----------------")
    # print(sorted(most_winns.items(), key=lambda item: item[1], reverse=True))

    # print("-----------------")

    triplet_comparisons = [(float(a), float(b)) for a, b in triplet_comparisons]
    return new_triplets, triplet_comparisons
