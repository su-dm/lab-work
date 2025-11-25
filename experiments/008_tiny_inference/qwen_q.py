from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    quantization_config=quantization_config,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

#outputs = model.generate(**inputs, max_new_tokens=40)
#print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

with open("/home/djole/code/lab-work/experiments/008_tiny_inference/uber_v_heller.txt", "r") as f:
    text = f.read()
    # calculate and print number of tokens in the text
    print("Tokens:",len(tokenizer.encode(text)))
    length = len(text)
    text = text[:length//2]
    messages = [
        {"role": "user", "content": "Summarize the following text: \n" + text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))