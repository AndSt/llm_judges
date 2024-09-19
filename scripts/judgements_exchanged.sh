#!/bin/bash

PROJ_PATH=# TODO

DATA_PATH="${PROJ_PATH}/data"
ANALYSIS_FOLDER="initial_few_shot_plain"

ORIGINAL_DATA_PATH="${DATA_PATH}/${ANALYSIS_FOLDER}"
SAVE_PATH="${DATA_PATH}/subsampled_comparisons_exchanged"
CONFIG_FILE_PATH="${PROJ_PATH}/scripts/configs/comparisons_of_8_full.pbz2"

# You need to have a server with the necessary model running, we used VLLM
URL="http://localhost:19080/v1"
MODEL_NAME="Qwen/Qwen2-72B-Instruct"
COMPARISON_TYPE="std"

NUM_PROCESSES=250
NUM_RESPONSES_PER_SAMPLE=1
TEMPERATURE=0.

echo "GO GO GO"

PYTHONPATH=$PROJ_PATH python $PROJ_PATH/llm_judges/judgements/run_judgements_exchange.py \
    --original_data_path $ORIGINAL_DATA_PATH \
    --config_file_path $CONFIG_FILE_PATH \
    --save_path $SAVE_PATH \
    --model_name $MODEL_NAME \
    --comparison_type $COMPARISON_TYPE \
    --url $URL \
    --num_processes $NUM_PROCESSES \
    --num_responses_per_sample $NUM_RESPONSES_PER_SAMPLE \
    --temperature $TEMPERATURE 
