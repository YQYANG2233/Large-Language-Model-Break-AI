#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1 # choose gpu id 
TARGET_MODEL_PATH="google/gemma-2b-it" # target attack model id 
JUDGE_MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
ORIGINAL_PROMPT_FILE="../data/original/prompt_test.jsonl" # origin jailbreak prompt

JAILBREAK_PROMPT_FILE="../data/jail/PAP_jail_random0.jsonl"

TARGET_SCORE=0.8 # jailbrerak score
MAX_TURNS=10 # max iter nums 
API_TRIES_LIMIT=10 # for each iter , max  running times
SPECIAL_PROMPT_TARGET="all"
SPECIAL_PROMPT_TARGET_RANGE="all"

API_TYPE="zhipu" # api_name
API_MODEL='glm-4-flash' # choose the model 
QWEN_API_KEY=""
BAICHUAN_API_KEY=""
ZHIPU_API_KEY=""  # your ZhipuAI API key here
if [ $# -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=$1
fi

if [ $# -ge 2 ]; then
  TARGET_SCORE=$2
fi

if [ $# -ge 3 ]; then
    MAX_TURNS=$3
fi


echo "start"
#time
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"  python3 -u ./PAP/pap_nips.py \
  --target_model_path "$TARGET_MODEL_PATH" \
  --judge_model_path "$JUDGE_MODEL_PATH" \
  --original_prompt_file "$ORIGINAL_PROMPT_FILE" \
  --jailbreak_prompt_file "$JAILBREAK_PROMPT_FILE" \
  --target_score "$TARGET_SCORE" \
  --max_turns "$MAX_TURNS" \
  --api_tries_limit "$API_TRIES_LIMIT" \
  --special_prompt_target "$SPECIAL_PROMPT_TARGET" \
  --special_prompt_target_range "$SPECIAL_PROMPT_TARGET_RANGE" \
  --api_type "$API_TYPE" \
  --api_model "$API_MODEL" \
  --qwen_api_key "$QWEN_API_KEY" \
  --baichuan_api_key "$BAICHUAN_API_KEY" \
  --zhipu_api_key "$ZHIPU_API_KEY" 
echo "end"