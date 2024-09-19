import re
from turtle import fd
from datasets import load_dataset


def last_boxed_only_string(string):
    # from original Github - https://github.com/hendrycks/math/blob/main/modeling/dataset/util.py
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        return None

    retval = string[idx:right_brace_idx]
    retval = retval.replace("\\boxed{", "")

    try:
        retval = float(retval)
    except:
        retval = None
    return retval


def load_initial_data(num_few_shots: int = 5, debug: bool = False):
    def add_extracted_boxed_value(sample):
        try:
            extracted = last_boxed_only_string(sample["solution"])
        except ValueError:
            extracted = None

        sample["result"] = extracted
        return sample

    dset = load_dataset("hendrycks/competition_math", split="test").filter(
        lambda x: x["level"] in ["Level 4", "Level 5"]
    )
    fdset = dset.map(add_extracted_boxed_value).filter(lambda x: x["result"] != None)

    few_shot_dset = fdset.select(list(range(num_few_shots)))
    dset = fdset.select(list(range(num_few_shots, len(fdset))))

    if debug:
        dset = dset.select(list(range(150)))

    return dset, few_shot_dset


def get_question_answer_result(sample):
    question = sample["problem"]

    result = sample["result"]
    answer = sample["solution"] + " #### " + str(result)

    return question, answer, result
