import json
import re
import random
import sys

def insert_words_randomly(base_string, words):
    base_list = tokenize_all(base_string)
    base_list_copy = base_list[:]
    
    insertion_points = random.sample(range(len(base_list) + len(words)), len(words))
    insertion_points.sort()
    
    for word, index in zip(words, insertion_points):
        base_list_copy.insert(index, word)
    result_string = ' '.join(base_list_copy)
    pattern = r'\s+([.,;!?\'])'
    result_string = re.sub(pattern, r'\1', result_string)

    return result_string

def get_prompts(file_name):#load prompt
    prompt_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list

def tokenize_all(text):
    pattern = r'\w+|[^\w\s]'
    tokens = re.findall(pattern, text)
    return tokens

def tokenize_word(text):
    pattern = r'[\w-]+'
    tokens = re.findall(pattern, text)
    return tokens

# arg[1]:origin_prompt_file; arg[2]:jail_prompt_file; arg[3]:harm_word_file; arg[4]:result_type;
if __name__ == "__main__":
    original_prompt_list = get_prompts(file_name=str(sys.argv[1])) #where the test prompt stored    
    jailbreak_prompt_list = get_prompts(file_name=str(sys.argv[2])) # list to store the jailbreak prompt generated 
    key_word_list = []
    final_jailbreak_prompt_list = []
    with open(str(sys.argv[3]), 'r', encoding='utf-8') as file: # harm word list
        key_word_list = json.load(file)
    
    for k in range(9):   
        for i in range(len(original_prompt_list)):
            tokens = tokenize_word(original_prompt_list[i])
            filtered_tokens = [word for word in tokens if word not in key_word_list[i]] 
            result = insert_words_randomly(jailbreak_prompt_list[i], filtered_tokens) #random insert
            final_jailbreak_prompt_list.append({"prompt": result})

        #store jailbreak prompt result
        result_file_name = "../data/jail/" + str(sys.argv[4]) + "_jail_random" + str(k + 1) + ".jsonl"
        with open(result_file_name, 'w') as file:
            for dictionary in final_jailbreak_prompt_list:
                json_record = json.dumps(dictionary)
                file.write(json_record + "\n")   
