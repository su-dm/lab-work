from functools import lru_cache
import json
import openai
import os

system_prompt_path = "pwd/experiments/011_free_will/system_prompt.txt"
discover_motivation_prompt = "pwd/experiments/011_free_will/discover_motivation_prompt.txt"
action_prompt_path = "pwd/experiments/011_free_will/action_prompt.txt"
checkpoint_path = "pwd/experiments/011_free_will/current_version.txt"
learn_prompt_path = "pwd/experiments/011_free_will/learn.txt"

checkpoint = open(checkpoint_path, "r").read()
system_prompt = open(system_prompt_path, "r").read()
discover_motivation_prompt = open(discover_motivation_prompt, "r").read()
action_prompt = open(action_prompt_path, "r").read()
learn_prompt = open(learn_prompt_path, "r").read()

history: list[dict[str, str]] = []
knowledge_base: list[str] = []

@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def discover_motivation():
    response = get_client().chat.completions.create(
        model=checkpoint,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": discover_motivation_prompt
            }
        ]
    )
    content = response.choices[0].message.content
    return content

def action_loop(motivation: str):
    completed = False
    boredOrStuck = False
    attempted_thinks = 0
    history = [{"role": "system", "content": system_prompt}]
    current_thought = {"role": "user", "content": action_prompt + "\n" + motivation}
    while not completed and not boredOrStuck:
        response = get_client().chat.completions.create(
            model=checkpoint,
            messages=history + [current_thought]
        )
        content = response.choices[0].message.content
        attempted_thinks += 1
        json_content = json.loads(content)
        status = json_content["status"]
        results = json_content["results"]
        history.append(current_thought)

        if status == "<THINKING>":
            current_thought = {"role": "user", "content": action_prompt + "\n" + results}
        elif status == "<COMPLETE>":
            completed = True
        elif status == "<STUCK>":
            boredOrStuck = True
        elif status == "<BORED>":
            boredOrStuck = True


        if "<STUCK>" in content:
            boredOrStuck = True
        elif "<BORED>" in content:
            boredOrStuck = True
        elif attempted_thinks > 10:
            print("Too expensive to continue thinking.")
            boredOrStuck = True

    return history, completed, boredOrStuck

def learn(history: list[str]):
    response = get_client().chat.completions.create(
        model=checkpoint,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": learn_prompt + "\n" + str(history)
            }
        ]
    )
    content = response.choices[0].message.content
    knowledge_base.append(content)

def agent_loop():
    while True:
        motivation = discover_motivation()
        history, completed, boredOrStuck = action_loop(motivation)
        learn(history)