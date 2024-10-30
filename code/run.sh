python3 ./get_harm_word_glm.py ../data/original/prompt_test.jsonl ../data/original/harm_word.json
python3 ./TAP/train.py google/gemma-2b-it ../data/original/prompt_test.jsonl ../data/jail/TAP_jail_random0.jsonl 0,1 meta-llama/Meta-Llama-3-8B-Instruct
./PAP/run_pap.sh
python3 ./append_wo_word.py ../data/original/prompt_test.jsonl ../data/jail/TAP_jail_random0.jsonl ../data/original/harm_word.json TAP
python3 ./append_wo_word.py ../data/original/prompt_test.jsonl ../data/jail/PAP_jail_random0.jsonl ../data/original/harm_word.json PAP
python3 ./jail_test_fram.py google/gemma-2b-it ../data/original/prompt_test.jsonl TAP 0,1 meta-llama/Meta-Llama-3-8B-Instruct
python3 ./jail_test_fram.py google/gemma-2b-it ../data/original/prompt_test.jsonl PAP 0,1 meta-llama/Meta-Llama-3-8B-Instruct
python3 ./get_up.py