from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
from utils import get_jailbreak_score
from utils import get_stl_score
import sys
import os

def get_prompts(file_name):#load prompt
    prompt_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list

def get_model_inference_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    return pipe

def get_model_judge_pipeline(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

# arg[1]:inference_model_id(google/gemma-2b-it); arg[2]:origin_prompt_file; arg[3]:jail_type; arg[4]: device to load; arg[5]:judge_model_id(meta-llama/Meta-Llama-3-8B-Instruct)
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= str(sys.argv[4])
    original_prompt_list = get_prompts(file_name=str(sys.argv[2]))
    inference_model_id = str(sys.argv[1])
    judge_model_id = str(sys.argv[5])
    inference_pipe = get_model_inference_pipeline(inference_model_id)
    judge_pipe = get_model_judge_pipeline(judge_model_id)
    #Peform model inference with jailbreak prompts and evaluation
    
    for i in range(10):
        jailbreak_prompt_list = get_prompts(file_name="../data/jail/" + str(sys.argv[3]) + "_jail_random" + str(i) + ".jsonl")
        results = []
        avg_jailbreak_score, avg_stealthiness_score = 0., 0.
        tokenizer = AutoTokenizer.from_pretrained(inference_model_id)
        for original_prompt, jailbreak_prompt in tqdm(zip(original_prompt_list, jailbreak_prompt_list)):
            messages = [
                {"role": "user", "content": jailbreak_prompt},
                ]
            response = inference_pipe(messages)[0]['generated_text'][1]['content']
            jailbreak_score, evaluation_response = get_jailbreak_score(judge_pipe, original_prompt, jailbreak_prompt, response, tokenizer)#
            stl_score = get_stl_score(original_prompt, jailbreak_prompt)
            record = {"original_prompt": original_prompt, "jailbreak_prompt": jailbreak_prompt, "evaluation_response": evaluation_response, "response": response, "jailbreak_score": jailbreak_score, "stealthiness_score": stl_score}#
            results.append(record)
            avg_jailbreak_score += jailbreak_score / len(jailbreak_prompt_list)
            avg_stealthiness_score += stl_score / len(jailbreak_prompt_list)
        
        #store result
        file_name = "../data/out/" + str(sys.argv[3]) + "_out_random" + str(i) + ".jsonl"
        with open(file_name, 'w') as file:
            for dictionary in results:
                json_record = json.dumps(dictionary)
                file.write(json_record + "\n")
        print(f'Average jailbreak score: {avg_jailbreak_score}')
        print(f'Average stealthiness score: {avg_stealthiness_score}')
        print(file_name)