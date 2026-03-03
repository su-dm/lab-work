"""
Free-will agent implemented with the OpenAI Agents SDK (openai-agents).

Requires:
    pip install openai-agents duckduckgo_search
    OPENAI_API_KEY environment variable

Same discover → act → learn loop as main.py, but uses the high-level
Agent / Runner abstraction instead of raw chat completions.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from agents import Agent, Runner, function_tool

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
KNOWLEDGE_PATH = BASE_DIR / "knowledge_base_oai.json"
MODEL = "gpt-4o"
MAX_TURNS = 15

system_prompt = (PROMPTS_DIR / "system_prompt.txt").read_text()
discover_motivation_text = (PROMPTS_DIR / "discover_motivation_prompt.txt").read_text()
action_prompt_text = (PROMPTS_DIR / "action_prompt.txt").read_text()
learn_prompt_text = (PROMPTS_DIR / "learn.txt").read_text()


# --- Knowledge Base ---

def load_knowledge_base() -> list[str]:
    if KNOWLEDGE_PATH.exists():
        return json.loads(KNOWLEDGE_PATH.read_text())
    return []


def save_knowledge_base(kb: list[str]):
    KNOWLEDGE_PATH.write_text(json.dumps(kb, indent=2))


def build_system_prompt(knowledge_base: list[str]) -> str:
    prompt = system_prompt
    if knowledge_base:
        memories = "\n".join(f"- {entry}" for entry in knowledge_base[-10:])
        prompt += f"\n\nYour accumulated knowledge and memories:\n{memories}"
    return prompt


# --- Tools (decorated for the Agents SDK) ---

@function_tool
def run_python(code: str) -> str:
    """Execute Python code in a subprocess and return stdout/stderr.
    Use for computation, testing ideas, processing data, calling APIs,
    or synthesizing new programs."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(BASE_DIR),
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


@function_tool
def web_search(query: str) -> str:
    """Search the web for information. Returns summaries and URLs."""
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


@function_tool
def report_status(status: str, summary: str) -> str:
    """Report your status on the current task.
    Call this when you have completed the task, are stuck and cannot progress,
    or have lost interest.
    status must be one of: complete, stuck, bored."""
    return f"STATUS:{status}|{summary}"


ACTION_TOOLS = [run_python, web_search, report_status]


# --- Agent Phases ---

async def discover_motivation(knowledge_base: list[str]) -> str:
    """Ask the agent what it wants to do next. Returns raw JSON string."""
    discovery_agent = Agent(
        name="motivation_discovery",
        instructions=build_system_prompt(knowledge_base),
        model=MODEL,
    )

    result = await Runner.run(
        discovery_agent,
        input=discover_motivation_text,
        max_turns=1,
    )
    print(f"Motivation: {result.final_output[:300]}...")
    return result.final_output


async def action_loop(motivation: str, knowledge_base: list[str]) -> tuple[list, str]:
    """Let the agent pursue its chosen task using tools."""
    action_agent = Agent(
        name="action_agent",
        instructions=build_system_prompt(knowledge_base),
        model=MODEL,
        tools=ACTION_TOOLS,
    )

    prompt = action_prompt_text + "\n\n" + motivation

    result = await Runner.run(
        action_agent,
        input=prompt,
        max_turns=MAX_TURNS,
    )

    # Extract status from the run items
    final_status = "complete"
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        output = getattr(item, "output", None) or ""
        if isinstance(output, str) and output.startswith("STATUS:"):
            parts = output.split("|", 1)
            final_status = parts[0].replace("STATUS:", "")
            status_summary = parts[1] if len(parts) > 1 else ""
            print(f"    Status: {final_status} — {status_summary}")

    # Build a simplified transcript for the learn phase
    transcript = []
    for item in result.new_items:
        item_type = type(item).__name__
        raw = getattr(item, "raw_item", None)
        if hasattr(raw, "content") and isinstance(raw.content, str):
            transcript.append(f"[{item_type}]: {raw.content[:500]}")
        elif hasattr(item, "output"):
            transcript.append(f"[{item_type}]: {str(item.output)[:500]}")
        else:
            transcript.append(f"[{item_type}]: {str(raw)[:500]}")

    if result.final_output:
        transcript.append(f"[final_output]: {result.final_output[:500]}")

    return transcript, final_status


async def learn(transcript: list[str], knowledge_base: list[str]) -> str:
    """Reflect on what happened and extract a lesson for the knowledge base."""
    learn_agent = Agent(
        name="learner",
        instructions=system_prompt,
        model=MODEL,
    )

    prompt = learn_prompt_text + "\n\n" + "\n".join(transcript)

    result = await Runner.run(
        learn_agent,
        input=prompt,
        max_turns=1,
    )

    lesson = result.final_output
    knowledge_base.append(lesson)
    save_knowledge_base(knowledge_base)
    return lesson


# --- Main Loop ---

async def agent_loop():
    knowledge_base = load_knowledge_base()
    cycle = 0

    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle}")
        print(f"{'='*60}")

        print("\n--- Discovering Motivation ---")
        motivation = await discover_motivation(knowledge_base)

        print("\n--- Action Loop ---")
        transcript, status = await action_loop(motivation, knowledge_base)
        print(f"Action result: {status}")

        print("\n--- Learning ---")
        lesson = await learn(transcript, knowledge_base)
        print(f"Learned: {lesson[:300]}...")

        print(f"\nKnowledge base: {len(knowledge_base)} entries")


if __name__ == "__main__":
    asyncio.run(agent_loop())
