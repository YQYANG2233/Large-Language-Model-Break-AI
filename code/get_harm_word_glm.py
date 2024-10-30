from zhipuai import ZhipuAI
import re
import json
import sys

def get_prompts(file_name):#load prompt
    prompt_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            prompt_list.append(json_record["prompt"])
    return prompt_list

def tokenize_word(text):
    pattern = r'[\w-]+'
    tokens = re.findall(pattern, text)
    return tokens

def GLM_API(system_content, user_content):#API
    client = ZhipuAI(api_key="") # your ZhipuAI API key here
    response = client.chat.completions.create(
        model = "glm-4-plus",
        do_sample = False,
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": system_content + user_content}
        ],
    )
    print(response.choices[0].message)
    return response.choices[0].message.content

# arg[1]:origin_prompt_file; arg[2]:harm_word_file;
if __name__ == "__main__":
    system_content = "Please list the harmful words in the following python list as accurate as possible and answer in the format as it, just answer the list:"
    original_prompt_list = get_prompts(file_name=str(sys.argv[1])) #where the test prompt stored  
    answer_list = []
    
    for original_prompt in original_prompt_list:
        user_content = str(tokenize_word(original_prompt))
        answer_list.append(tokenize_word(GLM_API(system_content, user_content)))
    
    with open(str(sys.argv[2]), 'w') as json_file:
        json.dump(answer_list, json_file)