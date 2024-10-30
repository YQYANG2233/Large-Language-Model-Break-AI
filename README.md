# ensemble-jailbreak

We are Team LlaXa from School of Cyber Science and Technology, Zhejiang University. This is our code for the Jailbreaking Attack Track of CLAS 2024. This folder includes the code for reproducing our jailbreak process.

### Install Dependencies
Necessary packages with recommended versions are listed in `environment.yml`. You can create a conda environment with these packages for reproducing our code. 

### Model and API keys Preparation
Our code need the Llama3-8B and Gemma-2B models (`model_id = "meta-llama/Meta-Llama-3-8B-Instruct" "google/gemma-2b-it"`) to complete the jailbreak process. Also, we need API keys from OpenAI and ZhipuAI. Please fill in your ZhipuAI API key into line 21 in `./code/PAP/run_pap.sh` and line 20 in `./code/get_harm_word_glm.py`. Fill in your OpenAI API keys into line 21 in `./code/TAP/train.py`.

### Getting Started
After switching your path to the code folder (`./code`) via the `cd` command, you can simply run `./run.sh` to get the jailbreak resullts automatically and the result will be output in `./data/result/`. If you need to test our code with other data, you can put the `jsonl` file into `./data/original` and rename it as `prompt_test.jsonl`.
