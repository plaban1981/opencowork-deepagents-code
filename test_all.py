#!/usr/bin/env python3
"""
Comprehensive functional test suite for opencowork-deepagents.
Tests: imports, tool validation, run_shell, fetch_url, search_web, agent creation, end-to-end task.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# ── Load .env so OPENCOWORK_API_KEY is set ──────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Helpers ──────────────────────────────────────────────────────────────────
PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []

def check(name: str, passed: bool, detail: str = ""):
    results.append((name, passed, detail))
    status = PASS if passed else FAIL
    line = f"{status} {name}"
    if detail:
        line += f"  →  {detail}"
    print(line)

def section(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ════════════════════════════════════════════════════════════════
# 1. DEPENDENCY IMPORTS
# ════════════════════════════════════════════════════════════════
section("1. Dependency Imports")

def test_imports():
    pkgs = [
        ("deepagents", "deepagents"),
        ("httpx", "httpx"),
        ("beautifulsoup4", "bs4"),
        ("ddgs", "ddgs"),
        ("python-dotenv", "dotenv"),
        ("langchain-core", "langchain_core"),
        ("langchain-anthropic", "langchain_anthropic"),
    ]
    for display, mod in pkgs:
        try:
            __import__(mod)
            check(f"import {display}", True)
        except ImportError as e:
            check(f"import {display}", False, str(e))

test_imports()


# ════════════════════════════════════════════════════════════════
# 2. TOOLS MODULE - UNIT TESTS (no network / no LLM)
# ════════════════════════════════════════════════════════════════
section("2. tools.py — Validation Logic")

import tools  # noqa: E402 (after sys.path is set)

def test_validate_command():
    # Should pass
    ok_cases = [
        "ls -la",
        "find . -name '*.txt' | head -10",
        "python script.py",
        "git status",
        "grep -r pattern . && echo done",
        "cat file.txt | sort | uniq",
        # semicolons inside quotes must NOT split the command (the fix)
        'python -c "import os; print(os.listdir(\'.\'))"',
        "python -c \"import sys; print(sys.version)\"",
        "bash -c \"echo hello; echo world\"",
    ]
    for cmd in ok_cases:
        valid, err = tools.validate_command(cmd)
        check(f"validate_command ALLOW: {cmd[:60]}", valid, err)

    # Should fail (blocked patterns)
    blocked_cases = [
        ("sudo ls", "sudo"),
        ("curl https://example.com", "curl"),
        ("wget http://x.com", "wget"),
        ("ssh user@host", "ssh"),
        ("chmod 777 file.txt", "chmod"),
        ("kill -9 1234", "kill"),
        ("cat /etc/passwd", "/etc/passwd"),
        ("ls ~/.ssh/", ".ssh"),
    ]
    for cmd, reason in blocked_cases:
        valid, err = tools.validate_command(cmd)
        check(f"validate_command BLOCK ({reason}): {cmd[:40]}", not valid, err)

    # Should fail (not in allowlist)
    not_allowed = ["nmap -sV localhost", "docker ps", "pip install requests"]
    for cmd in not_allowed:
        valid, err = tools.validate_command(cmd)
        check(f"validate_command NOT-ALLOWED: {cmd[:40]}", not valid, err)

test_validate_command()


# ── extract_base_commands — semicolon-in-quotes fix ──────────────────────────
section("2b. extract_base_commands — semicolon-in-quotes fix")

def test_extract_base_commands():
    cases = [
        # (command, expected_commands)
        ("ls -la", ["ls"]),
        ("cat file.txt | sort | uniq", ["cat", "sort", "uniq"]),
        ("grep pattern . && echo done", ["grep", "echo"]),
        # semicolons inside quotes must NOT produce extra entries
        ('python -c "import os; print(os.listdir(\'.\'))"', ["python"]),
        ('python -c "import sys; sys.exit(0)"', ["python"]),
        ("bash -c \"echo hello; echo world\"", ["bash"]),
        # env-var prefix skipped
        ("HOME=/tmp python script.py", ["python"]),
        # piped chain
        ("find . -name '*.txt' | head -10", ["find", "head"]),
    ]
    for cmd, expected in cases:
        got = tools.extract_base_commands(cmd)
        check(
            f"extract_base_commands: {cmd[:55]}",
            got == expected,
            f"expected={expected}  got={got}",
        )

test_extract_base_commands()


def test_validate_paths():
    sandbox = Path("C:/Temp/sandbox_test")
    tools.ALLOWED_DIRECTORIES = [sandbox]

    # Relative paths — no absolute path in command → should pass
    valid, err = tools.validate_paths_in_command("ls -la")
    check("validate_paths: relative cmd passes", valid, err)

    # Path inside sandbox
    valid, err = tools.validate_paths_in_command(f"cat {sandbox}/file.txt")
    check("validate_paths: in-sandbox path passes", valid, err)

    # Path outside sandbox
    valid, err = tools.validate_paths_in_command("cat /etc/hosts")
    check("validate_paths: out-of-sandbox /etc/hosts blocked", not valid, err)

    # Reset
    tools.ALLOWED_DIRECTORIES = []

test_validate_paths()


# ════════════════════════════════════════════════════════════════
# 3. run_shell TOOL — SANDBOX + WINDOWS TRANSLATION
# ════════════════════════════════════════════════════════════════
section("3. run_shell — Sandbox Enforcement + Windows Translation")

IS_WINDOWS = sys.platform == "win32"

def test_run_shell_sandbox():
    import tempfile, shutil

    # No directory granted yet
    tools.ALLOWED_DIRECTORIES = []
    result = tools.run_shell.invoke("ls")
    check("run_shell: no-grant returns error", "No directories" in result, result[:80])

    tmpdir = Path(tempfile.mkdtemp())
    tools.ALLOWED_DIRECTORIES = [tmpdir]
    (tmpdir / "hello.txt").write_text("hello world")
    (tmpdir / "subdir").mkdir()
    (tmpdir / "subdir" / "deep.txt").write_text("nested")

    # ls → translated to "dir /b" on Windows
    result = tools.run_shell.invoke("ls")
    check("run_shell: ls lists files (translated on Windows)", "hello.txt" in result, result[:120])

    # cat → translated to "type" on Windows
    result = tools.run_shell.invoke("cat hello.txt")
    check("run_shell: cat reads file content (translated on Windows)", "hello world" in result, result[:80])

    # mkdir -p → mkdir on Windows (parents created automatically)
    result = tools.run_shell.invoke("mkdir -p newdir/sub")
    check("run_shell: mkdir -p (translated on Windows)", "Blocked" not in result, result[:80])

    # find → translated to "dir /S /B *.txt" on Windows
    result = tools.run_shell.invoke("find . -name '*.txt'")
    check(
        "run_shell: find *.txt (translated on Windows)",
        "hello.txt" in result or ".txt" in result,
        result[:120],
    )

    # Python one-liner with semicolons inside quotes — now allowed (semicolon fix)
    result = tools.run_shell.invoke('python -c "import sys; print(sys.platform)"')
    check("run_shell: python -c with semicolon in quotes works", "Error" not in result and "Blocked" not in result, result[:80])

    # Blocked: sudo
    result = tools.run_shell.invoke("sudo ls")
    check("run_shell: sudo blocked", "Blocked" in result, result[:80])

    # Blocked: curl
    result = tools.run_shell.invoke("curl https://example.com")
    check("run_shell: curl blocked", "Blocked" in result, result[:80])

    # git status (works cross-platform)
    result = tools.run_shell.invoke("git status")
    check("run_shell: git status runs", "Blocked" not in result, result[:80])

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)
    tools.ALLOWED_DIRECTORIES = []

test_run_shell_sandbox()


# ── Windows translation unit tests ───────────────────────────────────────────
section("3b. Windows Command Translation")

def test_windows_translation():
    cases = [
        ("ls",             "dir"),
        ("ls -la",         "dir"),
        ("cat file.txt",   "type file.txt"),
        ("mkdir -p a/b",   "mkdir a/b"),
        ("find . -name '*.py'",  "dir /S /B *.py"),
        ('find . -name "*.txt"', "dir /S /B *.txt"),
        # No-op: python/git unchanged
        ("python script.py",  "python script.py"),
        ("git status",         "git status"),
    ]
    for unix_cmd, expected_fragment in cases:
        translated = tools.translate_for_windows(unix_cmd)
        check(
            f"translate_for_windows: {unix_cmd}",
            expected_fragment.lower() in translated.lower(),
            f"got: {translated}",
        )

test_windows_translation()


# ════════════════════════════════════════════════════════════════
# 4. fetch_url TOOL
# ════════════════════════════════════════════════════════════════
section("4. fetch_url — Web Content Extraction")

def test_fetch_url():
    # Invalid scheme
    result = tools.fetch_url.invoke("ftp://example.com")
    check("fetch_url: invalid scheme rejected", "Error" in result, result[:80])

    # Valid public URL
    result = tools.fetch_url.invoke("https://httpbin.org/get")
    check("fetch_url: https://httpbin.org/get returns content", len(result) > 50, result[:120])

    # Non-existent URL — should return HTTP Error
    result = tools.fetch_url.invoke("https://httpbin.org/status/404")
    check("fetch_url: 404 handled gracefully", "404" in result or "Error" in result, result[:80])

test_fetch_url()


# ════════════════════════════════════════════════════════════════
# 5. search_web TOOL
# ════════════════════════════════════════════════════════════════
section("5. search_web — DuckDuckGo Search")

def test_search_web():
    result = tools.search_web.invoke("LangChain DeepAgents Python")
    check(
        "search_web: returns results for LangChain query",
        len(result) > 100 and ("http" in result or "Error" in result),
        result[:150],
    )

    result = tools.search_web.invoke("Python asyncio tutorial")
    check(
        "search_web: returns results for Python asyncio",
        len(result) > 100 and "http" in result,
        result[:120],
    )

test_search_web()


# ════════════════════════════════════════════════════════════════
# 6. AGENT CREATION
# ════════════════════════════════════════════════════════════════
section("6. Agent Creation (create_opencowork_agent)")

def test_agent_creation():
    try:
        from agent import create_opencowork_agent, PROVIDER, MODEL_ID
        agent = create_opencowork_agent()
        check("agent creation succeeds", agent is not None, f"provider={PROVIDER}, model={MODEL_ID}")
        check("agent has ainvoke", hasattr(agent, "ainvoke"), type(agent).__name__)
    except Exception as e:
        check("agent creation", False, str(e))
        traceback.print_exc()

test_agent_creation()


# ════════════════════════════════════════════════════════════════
# 7. END-TO-END AGENT TASK (real LLM call)
# ════════════════════════════════════════════════════════════════
section("7. End-to-End Agent Task (LLM + tool use)")

async def test_e2e():
    import tempfile, shutil
    tmpdir = Path(tempfile.mkdtemp())
    tools.ALLOWED_DIRECTORIES = [tmpdir]

    # Create some files for the agent to work with
    (tmpdir / "notes.txt").write_text("Meeting notes:\n- Fix the login bug\n- Deploy by Friday\n- Review PR #42")
    (tmpdir / "data.csv").write_text("name,age\nAlice,30\nBob,25\nCarol,35")

    try:
        from agent import create_opencowork_agent
        from langchain_core.messages import HumanMessage

        agent = create_opencowork_agent()

        # Task 1: simple file listing
        print(f"\n{INFO} Running task: 'list all files in the directory'")
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="List all files in the current directory")]},
            {"recursion_limit": 60},
        )
        messages = result.get("messages", [])
        last = messages[-1].content if messages else ""
        if isinstance(last, list):
            last = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in last)
        check(
            "e2e task: list files — agent responds",
            len(last) > 20,
            last[:120],
        )

        # Task 2: read a file and summarize
        print(f"\n{INFO} Running task: 'read notes.txt and summarize'")
        result2 = await agent.ainvoke(
            {"messages": [HumanMessage(content="Read notes.txt and give me a one-line summary of the action items")]},
            {"recursion_limit": 60},
        )
        messages2 = result2.get("messages", [])
        last2 = messages2[-1].content if messages2 else ""
        if isinstance(last2, list):
            last2 = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in last2)
        check(
            "e2e task: read & summarize — agent responds",
            len(last2) > 20,
            last2[:150],
        )

    except Exception as e:
        check("e2e task", False, str(e))
        traceback.print_exc()
    finally:
        tools.ALLOWED_DIRECTORIES = []
        shutil.rmtree(tmpdir, ignore_errors=True)

asyncio.run(test_e2e())


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
section("SUMMARY")

total = len(results)
passed = sum(1 for _, p, _ in results if p)
failed = total - passed

print(f"\n  Total : {total}")
print(f"  {PASS} : {passed}")
print(f"  {FAIL} : {failed}")

if failed:
    print("\nFailed tests:")
    for name, p, detail in results:
        if not p:
            print(f"  - {name}  →  {detail}")

print()
sys.exit(0 if failed == 0 else 1)
