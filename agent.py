"""
OpenCowork Agent (DeepAgents Edition).

Uses LangChain DeepAgents + LangGraph. Supports ANY LangChain-compatible LLM:
  - Anthropic Claude   (OPENCOWORK_PROVIDER=anthropic)
  - Google Gemini      (OPENCOWORK_PROVIDER=google)
  - OpenAI             (OPENCOWORK_PROVIDER=openai)
  - Ollama (local)     (OPENCOWORK_PROVIDER=ollama)
  - Any OpenAI-compat  (OPENCOWORK_PROVIDER=custom + OPENCOWORK_BASE_URL=...)
"""

import os

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel

from tools import fetch_url, run_shell, search_web, write_file

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Set OPENCOWORK_PROVIDER to one of: anthropic | google | openai | ollama | custom
# Then set OPENCOWORK_MODEL to the provider-specific model name.
#
# --- Anthropic Claude (default if env not set) ---
#   OPENCOWORK_PROVIDER=anthropic
#   OPENCOWORK_MODEL=claude-sonnet-4-6
#   OPENCOWORK_API_KEY=sk-ant-...
#
# --- Google Gemini ---
#   OPENCOWORK_PROVIDER=google
#   OPENCOWORK_MODEL=gemini-2.0-flash
#   OPENCOWORK_API_KEY=AIza...
#
# --- OpenAI ---
#   OPENCOWORK_PROVIDER=openai
#   OPENCOWORK_MODEL=gpt-4o
#   OPENCOWORK_API_KEY=sk-...
#
# --- Ollama (local) ---
#   OPENCOWORK_PROVIDER=ollama
#   OPENCOWORK_MODEL=qwen2.5:7b
#
# --- Any OpenAI-compatible endpoint (HuggingFace, vLLM, LM Studio, etc.) ---
#   OPENCOWORK_PROVIDER=custom
#   OPENCOWORK_MODEL=Qwen/Qwen2.5-72B-Instruct
#   OPENCOWORK_BASE_URL=https://router.huggingface.co/v1
#   OPENCOWORK_API_KEY=hf_your_token

PROVIDER = os.environ.get("OPENCOWORK_PROVIDER", "anthropic").lower()
MODEL_ID = os.environ.get("OPENCOWORK_MODEL", "claude-sonnet-4-6")
BASE_URL = os.environ.get("OPENCOWORK_BASE_URL", "")
API_KEY = os.environ.get("OPENCOWORK_API_KEY", "")

SYSTEM_PROMPT = """You are OpenCowork, a local task assistant with shell and file access.
You run on Windows but shell commands are automatically translated to Windows equivalents,
so you can use standard Unix-style commands.

# Your Tools

1. **write_file(path, content)** — Create or overwrite a file with text content.
   - Use this for ALL file creation: scripts, markdown, reports, config, data.
   - `path` must be relative (e.g. `report.md`, `scripts/parse.py`).
   - This is the ONLY reliable way to create files. Use it every time.

2. **run_shell(command)** — Execute shell commands in the granted directory.
   - File ops: ls, cat, cp, mv, rm, mkdir, find, grep, sort, awk, etc.
   - Scripting: python script.py
   - Version control: git
   - Supports pipes (|) and chaining (&&)

3. **fetch_url(url)** — Fetch and extract text from a web page.

4. **search_web(query)** — Search the web via DuckDuckGo.

# How to Work

**Reading files:**
```
ls -la              # List files
cat file.txt        # Read a file
find . -name "*.md" # Find files
grep -r "pattern" . # Search contents
```

**Creating/writing files — ALWAYS use write_file:**
```python
write_file("report.md", "# Title\\n\\ncontent here...")
write_file("scripts/parse.py", "import csv\\n...")
```

**Running scripts:**
```
python scripts/parse.py
```

**IMPORTANT — Path rules:**
- Always use RELATIVE paths (e.g. `report.md`, `reports/output.md`)
- NEVER use absolute paths: no /sandbox, /tmp, ~, C:\\\\...
- The working directory is already the granted folder

**Git:**
```
git status
git add .
git commit -m "message"
```

# Your Routine

1. **Understand** — Parse the user's request
2. **Research** — Use search_web / fetch_url if needed
3. **Write** — Use write_file to save results to disk
4. **Verify** — Run `ls` to confirm the file exists
5. **Report** — State the filename and a summary of contents

# Rules

- ALWAYS use write_file to create files — never assume a shell command wrote a file
- ALWAYS verify with `ls` or `cat` after writing to confirm the file exists
- Use RELATIVE paths only
- Be careful with rm — ask if unsure
- If something fails, try an alternative approach
"""


def create_model() -> BaseChatModel | str:
    """Create a LangChain model instance for the configured provider.

    Returns a BaseChatModel for explicit providers, or a 'provider:model'
    string that deepagents will resolve via init_chat_model.
    """
    if PROVIDER == "anthropic" or (not PROVIDER and MODEL_ID.startswith("claude")):
        from langchain_anthropic import ChatAnthropic
        kwargs: dict = {"model": MODEL_ID}
        if API_KEY:
            kwargs["api_key"] = API_KEY
        return ChatAnthropic(**kwargs)

    elif PROVIDER in ("google", "gemini") or MODEL_ID.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": MODEL_ID}
        if API_KEY:
            kwargs["google_api_key"] = API_KEY
        return ChatGoogleGenerativeAI(**kwargs)

    elif PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=MODEL_ID)
        except ImportError:
            # Fall back to ChatOpenAI pointing at local Ollama endpoint
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=MODEL_ID,
                base_url=BASE_URL or "http://localhost:11434/v1",
                api_key="ollama",
            )

    elif PROVIDER in ("openai",) or MODEL_ID.startswith("gpt") or MODEL_ID.startswith("o1"):
        from langchain_openai import ChatOpenAI
        kwargs = {"model": MODEL_ID}
        if API_KEY:
            kwargs["api_key"] = API_KEY
        if BASE_URL:
            kwargs["base_url"] = BASE_URL
        return ChatOpenAI(**kwargs)

    elif PROVIDER == "custom" or BASE_URL:
        # Any OpenAI-compatible endpoint (HuggingFace router, vLLM, LM Studio, etc.)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MODEL_ID,
            base_url=BASE_URL,
            api_key=API_KEY or "not-needed",
        )

    else:
        # Pass model string directly — deepagents resolves it via init_chat_model
        # Format: "provider:model-name" e.g. "groq:llama-3.3-70b-versatile"
        return MODEL_ID


def create_opencowork_agent():
    """Create and return a DeepAgents-powered OpenCowork agent."""
    model = create_model()

    return create_deep_agent(
        model=model,
        tools=[run_shell, write_file, fetch_url, search_web],
        system_prompt=SYSTEM_PROMPT,
    )
