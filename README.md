# From Calculation to Adjudication: Examining LLM Judges on Mathematical Reasoning Tasks

This repository contains code and resources related to the paper "From Calculation to Adjudication: Examining LLM Judges on Mathematical Reasoning Tasks". The paper investigates the use of Large Language Models (LLMs) as judges for evaluating the correctness of solutions to mathematical reasoning tasks.

Authors: Andreas Stephan, Dawei Zhu, Matthias Aßenmacher, Xiaoyu Shen, Benjamin Roth

For any questions, comments, or if you want to chat about anything related to this work, contact [andreas.stephan@univie.ac.at](mailto:andreas.stephan@univie.ac.at).

## Abstract

To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. LLM judges are typically evaluated by measuring the correlation with human judgments on generation tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that the used judges are mostly unable to improve task performance but are able to pick the better model. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance. We observe that judges tend to choose the model of higher quality even if its answer is incorrect. Further, we show that it is possible to use statistics, such as the task performances of the individual models, to predict judgment performance. In an ablation, we either swap or mask the candidate answers and observe that judges often keep the original judgment, providing evidence that judges incorporate writing style in their judgments. In summary, we find that regularities in the judgments are quantifiable using statistical measures and provide various angles on exploiting them.


## Key Findings

* **Larger Models as Better Judges:** The research confirms that larger LLMs generally outperform smaller ones in judging the correctness of solutions.
* **Correlation Between Judgment and Task Performance:** There is a strong correlation between the performance of the LLM judge and the task performance of the candidate models it's evaluating.
* **Bias Towards Larger Models:**  LLM judges tend to favor answers from stonger models, even when incorrect, suggesting a reliance on writing style and linguistic cues.
* **Predictability of Judgments:** The study demonstrates that judgment performance and individual judgments can be predicted to a certain extent using statistical measures and machine learning models.
* 
## Installation

There are two requirement files. If you want to explicitly work with our versions, run ```pip install fixed_requirements.txt```.
In case you want to work with newer version of the used libraries, run ```pip install requirements.txt```. Note that we do not continuously test whether this works.

Finally you can run ```pip install -e .``` to make sure imports work.

## Datasets and Models Evaluated

#### Datasets

* **AQUA-RAT:** A multiple-choice dataset testing quantitative reasoning.
* **GSM8K:** A dataset of grade school math word problems with free-form numerical answers.
* **MATH:** A dataset containing challenging competition mathematics problems.

#### Models Evaluated

* **Large Models:** Qwen-2 72B, Llama 3 70B, Yi 1.5 34B, Mixtral 8x7B
* **Small Models:** Llama 3 8B, Gemma 1.1 7B, Mistral 7B v0.3, Mistral 7B v0.1

## Run generations

In ```candidate_answers/run_candidates_few_shot.py``` you, find code to generate candidate answers for the used datasets. In ```scripts/candidate_answers.sh``` you find a bash file to run the file.

Similarly, you find the code for making judgements in ```judgements/``` and an example bash scripts for generating judements in ```scripts/judgements.sh```. There are also two bash scripts for our ablations where we mask (```_xxx```) numbers and where we swap results (```_exchanged```).

!!Important!! In each file you need to set the ```PROJ_PATH``` environment variable and it should point to the location of the repository.

## Citation

If you find this work useful, please cite the paper:


```
@misc{stephan2024calculationadjudicationexaminingllm,
      title={From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks}, 
      author={Andreas Stephan and Dawei Zhu and Matthias Aßenmacher and Xiaoyu Shen and Benjamin Roth},
      year={2024},
      eprint={2409.04168},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.04168}, 
}
```