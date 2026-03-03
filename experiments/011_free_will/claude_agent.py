"""
Free-will agent implemented with the Claude Agent SDK (claude-agent-sdk).

Requires:
    pip install claude-agent-sdk duckduckgo_search
    ANTHROPIC_API_KEY environment variable

Same discover → act → learn loop as main.py, but uses the Claude Agent SDK
which provides Claude Code's built-in tools and agent loop, plus custom
tools defined as in-process MCP servers.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    tool,
    create_sdk_mcp_server,
)

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
KNOWLEDGE_PATH = BASE_DIR / "knowledge_base_claude.json"
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


# --- Custom Tools (via in-process MCP server) ---

@tool("run_python", "Execute Python code in a subprocess and return stdout/stderr. "
      "Use for computation, testing ideas, processing data, calling APIs, "
      "or synthesizing new programs.", {"code": str})
async def run_python_tool(args):
    code = args["code"]
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
        text = output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        text = "Error: execution timed out after 30 seconds"
    except Exception as e:
        text = f"Error executing code: {e}"
    return {"content": [{"type": "text", "text": text}]}


@tool("web_search", "Search the web for information. Returns summaries and URLs.",
      {"query": str})
async def web_search_tool(args):
    query_str = args["query"]
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query_str, max_results=5))
        if not results:
            text = "No results found."
        else:
            formatted = []
            for r in results:
                formatted.append(f"**{r['title']}**\n{r['body']}\nURL: {r['href']}")
            text = "\n\n---\n\n".join(formatted)
    except ImportError:
        text = "Error: duckduckgo_search not installed. Run: pip install duckduckgo_search"
    except Exception as e:
        text = f"Search error: {e}"
    return {"content": [{"type": "text", "text": text}]}


@tool("report_status",
      "Report your status on the current task. Call this when you have "
      "completed the task, are stuck and cannot progress, or have lost interest. "
      "status must be one of: complete, stuck, bored.",
      {"status": str, "summary": str})
async def report_status_tool(args):
    status = args["status"]
    summary = args["summary"]
    text = f"STATUS:{status}|{summary}"
    return {"content": [{"type": "text", "text": text}]}


# Build the MCP server with our custom tools
tools_server = create_sdk_mcp_server(
    name="free-will-tools",
    version="1.0.0",
    tools=[run_python_tool, web_search_tool, report_status_tool],
)


# --- Helper: collect text from query stream ---

async def collect_response(prompt: str, options: ClaudeAgentOptions) -> str:
    """Run a query and collect all assistant text from the stream."""
    parts = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
        elif hasattr(message, "result"):
            parts.append(str(message.result))
    return "".join(parts)


# --- Agent Phases ---

async def discover_motivation(knowledge_base: list[str]) -> str:
    """Ask the agent what it wants to do next. Returns raw text/JSON."""
    sys_prompt = build_system_prompt(knowledge_base)
    options = ClaudeAgentOptions(
        system_prompt=sys_prompt,
        max_turns=1,
    )
    result = await collect_response(discover_motivation_text, options)
    return result


async def action_loop(motivation: str, knowledge_base: list[str]) -> tuple[str, str]:
    """Let the agent pursue its chosen task using custom tools.

    The Claude Agent SDK handles the tool-call loop internally.
    We provide our custom tools via an in-process MCP server.
    """
    sys_prompt = build_system_prompt(knowledge_base)
    prompt = action_prompt_text + "\n\n" + motivation

    options = ClaudeAgentOptions(
        system_prompt=sys_prompt,
        max_turns=MAX_TURNS,
        mcp_servers={"tools": tools_server},
        allowed_tools=[
            "mcp__tools__run_python",
            "mcp__tools__web_search",
            "mcp__tools__report_status",
        ],
        permission_mode="bypassPermissions",
    )

    # Collect the full transcript and watch for status reports
    transcript_parts = []
    final_status = "stuck"

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text = block.text
                    transcript_parts.append(f"[assistant]: {text[:500]}")
                    print(f"    {text[:200]}...")

                    # Check for status report in the response
                    if "STATUS:" in text:
                        for line in text.split("\n"):
                            if line.startswith("STATUS:"):
                                parts = line.split("|", 1)
                                final_status = parts[0].replace("STATUS:", "")
                                status_summary = parts[1] if len(parts) > 1 else ""
                                print(f"    Status: {final_status} — {status_summary}")
        elif hasattr(message, "result"):
            result_text = str(message.result)
            transcript_parts.append(f"[result]: {result_text[:500]}")

    transcript = "\n".join(transcript_parts)
    return transcript, final_status


async def learn(transcript: str, knowledge_base: list[str]) -> str:
    """Reflect on what happened and extract a lesson for the knowledge base."""
    prompt = learn_prompt_text + "\n\n" + transcript

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=1,
    )

    lesson = await collect_response(prompt, options)
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
        print(f"Motivation: {motivation[:300]}...")

        print("\n--- Action Loop ---")
        transcript, status = await action_loop(motivation, knowledge_base)
        print(f"Action result: {status}")

        print("\n--- Learning ---")
        lesson = await learn(transcript, knowledge_base)
        print(f"Learned: {lesson[:300]}...")

        print(f"\nKnowledge base: {len(knowledge_base)} entries")


if __name__ == "__main__":
    asyncio.run(agent_loop())
