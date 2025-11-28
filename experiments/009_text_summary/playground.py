import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="",
    #api_key=os.environ["HF_TOKEN"],
)

with open("experiments/009_text_summary/uber_v_heller.txt", "r") as f:
    text = f.read()
    text = text[:len(text)//2]

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507:nscale",
        messages=[
            {
                "role": "user",
                "content": "Provide a case brief of the following case:\n" + text
            }
        ],
    )

    print(completion.choices[0].message)
