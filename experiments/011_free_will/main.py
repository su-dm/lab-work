from functools import lru_cache
from pathlib import Path
import json
import subprocess
import sys
import openai
import os
import traceback

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
KNOWLEDGE_PATH = BASE_DIR / "knowledge_base.json"
MODEL = "gpt-5.2"
MAX_TURNS = 15

system_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text()
discover_motivation_text = (PROMPTS_DIR / "discover_motivation_prompt.txt").read_text()
action_prompt = (PROMPTS_DIR / "action_prompt.txt").read_text()
learn_prompt = (PROMPTS_DIR / "learn.txt").read_text()


# --- Knowledge Base (persisted to disk) ---

def load_knowledge_base() -> list[str]:
    if KNOWLEDGE_PATH.exists():
        return json.loads(KNOWLEDGE_PATH.read_text())
    return []

def save_knowledge_base(kb: list[str]):
    KNOWLEDGE_PATH.write_text(json.dumps(kb, indent=2))


# --- Tool Definitions (passed to the OpenAI API) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in a subprocess and return stdout/stderr. "
                "Use for computation, testing ideas, processing data, calling APIs, "
                "or synthesizing new programs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns summaries and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_status",
            "description": (
                "Report your status on the current task. Call this when you have "
                "completed the task, are stuck and cannot progress, or have lost interest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["complete", "stuck", "bored"],
                        "description": "Your current status"
                    },
                    "summary": {
                        "type": "string",
                        "description": "What you accomplished, or why you are stuck/bored"
                    }
                },
                "required": ["status", "summary"]
            }
        }
    }
]


# --- Tool Implementations ---

def run_python(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(BASE_DIR)
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\nProcess exited with code {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out after 30 seconds"
    except Exception as e:
        return f"Error executing code: {e}"


def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"**{r['title']}**\n{r['body']}\nURL: {r['href']}")
        return "\n\n---\n\n".join(formatted)
    except ImportError:
        return "Error: duckduckgo_search not installed. Run: pip install duckduckgo_search"
    except Exception as e:
        return f"Search error: {e}"


def report_status(status: str, summary: str) -> str:
    return f"Status reported: {status}. {summary}"


TOOL_REGISTRY = {
    "run_python": lambda **kw: run_python(kw["code"]),
    "web_search": lambda **kw: web_search(kw["query"]),
    "report_status": lambda **kw: report_status(kw["status"], kw["summary"]),
}


def execute_tool(name: str, arguments: dict) -> str:
    if name not in TOOL_REGISTRY:
        return f"Error: unknown tool '{name}'"
    try:
        return TOOL_REGISTRY[name](**arguments)
    except Exception as e:
        return f"Error in {name}: {e}\n{traceback.format_exc()}"


# --- Client ---

@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Agent Phases ---

def build_system_prompt(knowledge_base: list[str]) -> str:
    prompt = system_prompt
    if knowledge_base:
        memories = "\n".join(f"- {entry}" for entry in knowledge_base[-10:])
        prompt += f"\n\nYour accumulated knowledge and memories:\n{memories}"
    return prompt


def discover_motivation(knowledge_base: list[str]) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(knowledge_base)},
            {"role": "user", "content": discover_motivation_text}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


def action_loop(motivation: str, knowledge_base: list[str]) -> tuple[list[dict], str]:
    messages = [
        {"role": "system", "content": build_system_prompt(knowledge_base)},
        {"role": "user", "content": action_prompt + "\n\n" + motivation}
    ]

    turns = 0
    final_status = "stuck"
    consecutive_text_replies = 0

    while turns < MAX_TURNS:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message
        messages.append(msg.to_dict())
        turns += 1

        print(f"  [Turn {turns}/{MAX_TURNS}] finish_reason={response.choices[0].finish_reason}")

        if msg.tool_calls:
            consecutive_text_replies = 0
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f"    -> {fn_name}({', '.join(f'{k}=...' for k in fn_args)})")

                result = execute_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

                if fn_name == "report_status":
                    final_status = fn_args["status"]
                    print(f"    <- Status: {final_status} — {fn_args.get('summary', '')}")
                    return messages, final_status

                print(f"    <- {result[:200]}{'...' if len(result) > 200 else ''}")
        else:
            consecutive_text_replies += 1
            if msg.content:
                print(f"    Thinking: {msg.content[:200]}...")

            if consecutive_text_replies >= 2:
                messages.append({
                    "role": "user",
                    "content": (
                        "Use your available tools to take concrete action, "
                        "or call report_status to signal you are done."
                    )
                })

    print(f"  Hit max turns ({MAX_TURNS}).")
    return messages, final_status


def learn(history: list[dict], knowledge_base: list[str]) -> str:
    transcript = "\n".join(
        f"[{m.get('role', '?')}]: {(m.get('content') or '(tool call)')[:500]}"
        for m in history
        if m.get("role") != "system"
    )
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": learn_prompt + "\n\n" + transcript}
        ]
    )
    lesson = response.choices[0].message.content
    knowledge_base.append(lesson)
    save_knowledge_base(knowledge_base)
    return lesson


# --- Main ---

def agent_loop():
    knowledge_base = load_knowledge_base()
    cycle = 0

    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle}")
        print(f"{'='*60}")

        print("\n--- Discovering Motivation ---")
        motivation = discover_motivation(knowledge_base)
        print(f"Motivation: {motivation[:300]}...")

        print("\n--- Action Loop ---")
        history, status = action_loop(motivation, knowledge_base)
        print(f"Action result: {status}")

        print("\n--- Learning ---")
        lesson = learn(history, knowledge_base)
        print(f"Learned: {lesson[:300]}...")

        print(f"\nKnowledge base: {len(knowledge_base)} entries")


if __name__ == "__main__":
    agent_loop()
