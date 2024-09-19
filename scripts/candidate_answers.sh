#!/bin/bash

PROJ_PATH=# TODO

DATA_PATH="${PROJ_PATH}/data"

# You need to have a server with the necessary model running, we used VLLM
URL="http://localhost:19082/v1"

MODEL_NAME="01-ai/Yi-1.5-34B-Chat-16K"
DATASET_NAME="math"


NUM_PROCESSES=80
NUM_RESPONSES_PER_SAMPLE=11
TEMPERATURE=0.9

DEBUG=False

echo "GO GO GO"

PYTHONPATH=$PROJ_PATH python $PROJ_PATH/llm_judges/candidate_answers/run_candidates_few_shot.py \
    --data_path $DATA_PATH \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --url $URL \
    --num_processes $NUM_PROCESSES \
    --num_responses_per_sample $NUM_RESPONSES_PER_SAMPLE \
    --temperature $TEMPERATURE \
    # --debug 

echo "DONE"
