from datasets import load_dataset


def load_initial_data(num_few_shots: int=5, debug: bool=False):
    
    dset = load_dataset("aqua_rat", split="test")
    few_shot_dset = load_dataset("aqua_rat", split="train[0:20]").select(list(range(num_few_shots)))
    
    if debug:
        dset = dset.select(list(range(150)))
    
    return dset, few_shot_dset


def get_question_answer_result(sample):
    question = f"{sample['question']}\nOptions:\n"
    question += "\n".join(sample["options"])

    result = sample["correct"]
    answer = sample["rationale"] + " #### " + result

    return question, answer, result
