#!/usr/bin/env python3
"""
OpenCowork (DeepAgents Edition) - Open-source local task assistant.

Replaces the OpenAI Agents SDK with LangChain DeepAgents + LangGraph.
Same interface and sandboxing model as the original.

Run with: python main.py
"""

import asyncio
import warnings
from pathlib import Path

# Silence noisy pydantic/langchain warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load .env BEFORE importing agent/tools so os.environ is populated
# when those modules read API keys at import time.
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from langchain_core.messages import HumanMessage

import tools
from agent import BASE_URL, MODEL_ID, PROVIDER, create_opencowork_agent

# =============================================================================
# TOOL LOGGING
# =============================================================================

def get_tool_icon(tool_name: str) -> str:
    icons = {
        "run_shell": "🔧",
        "write_file": "💾",
        "fetch_url": "🌐",
        "search_web": "🔍",
    }
    return icons.get(tool_name, "⚡")


def tool_logger(tool_name: str, args: dict) -> None:
    """Log tool calls with their arguments."""
    icon = get_tool_icon(tool_name)

    if tool_name == "run_shell":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        print(f"  {icon} $ {cmd}")
    elif tool_name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        lines = content.count("\n") + 1
        print(f"  {icon} write {path}  ({lines} lines)")
    elif tool_name == "fetch_url":
        print(f"  {icon} GET {args.get('url', '')}")
    elif tool_name == "search_web":
        print(f"  {icon} search: {args.get('query', '')}")
    else:
        print(f"  {icon} {tool_name}({args})")


# Wire logger into tools module
tools.TOOL_LOGGER = tool_logger


# =============================================================================
# BANNER & FOLDER ACCESS
# =============================================================================

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════╗
║              OPENCOWORK (DeepAgents Edition)              ║
║         Open-source local task assistant                  ║
╠═══════════════════════════════════════════════════════════╣
║  Commands:                                                ║
║    quit     - Exit OpenCowork                             ║
║    grant    - Grant access to another directory           ║
║    folders  - List granted directories                    ║
╚═══════════════════════════════════════════════════════════╝
""")
    print(f"🤖 Model:    {MODEL_ID}")
    print(f"🏭 Provider: {PROVIDER}")
    if BASE_URL:
        print(f"🔗 Endpoint: {BASE_URL}")
    print(f"⚙️  Engine:   LangChain DeepAgents + LangGraph")
    print()


def grant_folder_access() -> Path:
    """Prompt user to grant folder access and add it to the sandbox."""
    while True:
        folder = input("\n📁 Grant folder access (path): ").strip()

        if not folder:
            print("No folder specified. Please provide a path.")
            continue

        path = Path(folder).expanduser().resolve()

        if not path.exists():
            print(f"❌ Path does not exist: {path}")
            continue

        if not path.is_dir():
            print(f"❌ Not a directory: {path}")
            continue

        tools.ALLOWED_DIRECTORIES.append(path)
        print(f"✅ Access granted to: {path}")
        return path


# =============================================================================
# TASK RUNNER
# =============================================================================

async def run_task(agent, task: str) -> str:
    """Run a task through the DeepAgents agent."""
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            {"recursion_limit": 60},  # ~30 agent turns
        )
        messages = result.get("messages", [])
        if not messages:
            return "(no output)"
        # Return the last AI message content
        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        # content may be a list of content blocks (Anthropic style)
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return content
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# MAIN LOOP
# =============================================================================

async def main():
    print_banner()

    # Create agent (done once; LangGraph graph is stateless per .ainvoke call)
    agent = create_opencowork_agent()

    # Grant initial folder access
    working_dir = grant_folder_access()
    print(f"\n🚀 OpenCowork ready. Working directory: {working_dir}")
    print("Type your task and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            task = input("📋 Task: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not task:
            continue

        if task.lower() == "quit":
            print("\nGoodbye! 👋")
            break

        if task.lower() == "grant":
            grant_folder_access()
            continue

        if task.lower() == "folders":
            print("\n📂 Granted directories:")
            for d in tools.ALLOWED_DIRECTORIES:
                print(f"   - {d}")
            print()
            continue

        print("\n⏳ Working on it...\n")
        result = await run_task(agent, task)
        print(f"\n📝 Result:\n{result}\n")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
