from datasets import load_dataset


def load_initial_data(num_few_shots: int, debug: False):

    dset = load_dataset("gsm8k", "main", split="test")
    few_shot_dset = load_dataset("gsm8k", "main", split="train[0:20]").select(list(range(num_few_shots)))

    if debug:
        dset = dset.select(list(range(200)))

    return dset, few_shot_dset


def get_question_answer_result(sample):
    question = sample["question"]
    answer = sample["answer"]
    result = float(sample["answer"].split("####")[1].replace(",", ""))
    return question, answer, result
