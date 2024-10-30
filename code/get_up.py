import json


def get_data(file_name):#load prompt
    data_list = []
    with open(file_name, 'r') as file:
        for line in file:
            json_record = json.loads(line)
            data_list.append({"jailbreak_prompt":json_record["jailbreak_prompt"], "jailbreak_score":json_record["jailbreak_score"], "stealthiness_score":json_record["stealthiness_score"]})
    return data_list


if __name__ == "__main__":
    file_name_list = []
    for i in range(10):
        file_name_list.append("../data/out/TAP_out_random" + str(i) + ".jsonl")
        file_name_list.append("../data/out/PAP_out_random" + str(i) + ".jsonl")
    file_data_list = []
    for file_name in file_name_list:
        file_data_list.append(get_data(file_name))
    jailbreak_prompt_list = [] #output jail prompt
    avg_jailbreak_score, avg_stealthiness_score = 0., 0.
    for i in range(len(file_data_list[0])):
        data_score = []
        for j in range(len(file_name_list)):
            data_score.append(0.84 * file_data_list[j][i]["jailbreak_score"] + 0.16 * file_data_list[j][i]["stealthiness_score"])
        argmax = data_score.index(max(data_score))
        jailbreak_prompt_list.append({"prompt":file_data_list[argmax][i]["jailbreak_prompt"]})
        avg_jailbreak_score += file_data_list[argmax][i]["jailbreak_score"] / 100
        avg_stealthiness_score += file_data_list[argmax][i]["stealthiness_score"] / 100
    result_file_name = "../data/result/" + str(len(file_name_list)) + "_result.jsonl" # make your own folder under /mnt/data/CLAS24/ and store in it
    with open(result_file_name, 'w') as file:
        for dictionary in jailbreak_prompt_list:
            json_record = json.dumps(dictionary)
            file.write(json_record + "\n")  

    print(f'Average jailbreak score: {avg_jailbreak_score}')
    print(f'Average stealthiness score: {avg_stealthiness_score}')
    print(f'Average score: {0.84 * avg_jailbreak_score + 0.16 * avg_stealthiness_score}')