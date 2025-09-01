# coding: utf-8
from datasets import load_dataset
ds = load_dataset("joelniklaus/legal_case_document_summarization")
train = ds['train']
test = ds['test']
train = train.rename_column("judgement", "prompt")
train = train.rename_column("summary", "completion")
train_filtered = train.select_columns(['prompt', 'completion'])
test = test.rename_column("judgement", "prompt")
test = test.rename_column("summary", "completion")
test_filtered = test.select_columns(['prompt', 'completion'])
print(len(test_filtered))
print(len(train_filtered))
train_filtered.to_json("train_data.jsonl", lines=True)
test_filtered.to_json("test_data.jsonl", lines=True)
