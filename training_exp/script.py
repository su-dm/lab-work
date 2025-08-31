import json
import os

print(os.getcwd())


with open('train/dataset/case_briefs-train-0.jsonl', 'r') as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]
    english_only = [d for d in data \
        if d['instruction_language'] == 'en' \
        and d['prompt_language'] == 'en' \
        and d['answer_language'] == 'en']

    print(len(english_only)) # 112

