"""
OpenCowork Tools (DeepAgents Edition) - Sandboxed shell access and web operations.

Same 3 tools as the original, redefined with LangChain's @tool decorator:
- run_shell: Allowlisted shell commands (file ops, search, scripting, git)
- fetch_url: Web content extraction
- search_web: Research queries via DuckDuckGo
"""

import os
import platform
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable

IS_WINDOWS = platform.system() == "Windows"

import httpx
from langchain_core.tools import tool

# =============================================================================
# SANDBOX CONFIGURATION
# =============================================================================

ALLOWED_DIRECTORIES: list[Path] = []

# Logging callback - set from main.py to log tool calls
TOOL_LOGGER: Callable | None = None


def log_tool_call(tool_name: str, args: dict) -> None:
    """Log a tool call if logger is set."""
    if TOOL_LOGGER:
        TOOL_LOGGER(tool_name, args)


# Allowlisted base commands (safer than blocklist)
ALLOWED_COMMANDS = {
    # File operations
    "ls", "cat", "head", "tail", "less", "more",
    "cp", "mv", "rm", "mkdir", "rmdir", "touch",
    # Search
    "find", "grep", "rg", "fd", "locate",
    # Text processing
    "sort", "uniq", "wc", "cut", "tr", "sed", "awk", "xargs",
    # Utilities
    "echo", "printf", "date", "stat", "file", "du", "df", "basename", "dirname",
    "true", "false", "test", "[",
    # Scripting (sandboxed to cwd)
    "python", "python3", "bash", "sh",
    # Archive
    "tar", "zip", "unzip", "gzip", "gunzip", "bzip2", "xz",
    # JSON/data
    "jq",
    # Version control
    "git",
    # Diff/patch
    "diff", "patch",
}

# Explicitly blocked patterns (defense in depth)
BLOCKED_PATTERNS = [
    # Privilege escalation
    r"\bsudo\b",
    r"\bsu\b",
    # Network tools (use fetch_url/search_web instead)
    r"\bcurl\b",
    r"\bwget\b",
    r"\bnc\b",
    r"\bnetcat\b",
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b",
    # Permission changes
    r"\bchmod\b",
    r"\bchown\b",
    # Process control
    r"\bkill\b",
    r"\bpkill\b",
    r"\bkillall\b",
    # Sensitive system paths
    r"/dev/",
    r"/proc/",
    r"/sys/",
    r"/etc/passwd",
    r"/etc/shadow",
    r"~/.ssh",
    r"\.ssh/",
    # Path traversal attempts
    r"\.\./",
    r"/\.\.",
]


_SHELL_OPERATORS = frozenset({"|", "||", "&&", ";", "&"})


def _first_cmd(segment: list[str]) -> str | None:
    """Return the command name from a tokenized segment, skipping env assignments."""
    for token in segment:
        if not token.startswith('-') and '=' not in token:
            return token
    return None


def extract_base_commands(command: str) -> list[str]:
    """Extract all base command names from a shell command string.

    Uses shlex.split so that semicolons/pipes *inside* quoted strings
    (e.g. python -c "import os; print(...)") are NOT treated as separators.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    commands: list[str] = []
    segment: list[str] = []
    for token in tokens:
        if token in _SHELL_OPERATORS:
            cmd = _first_cmd(segment)
            if cmd:
                commands.append(cmd)
            segment = []
        else:
            segment.append(token)
    cmd = _first_cmd(segment)
    if cmd:
        commands.append(cmd)
    return commands


def validate_command(command: str) -> tuple[bool, str]:
    """Validate a shell command against allowlist and blocklist."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Blocked pattern detected: {pattern}"

    base_commands = extract_base_commands(command)
    if not base_commands:
        return False, "No command detected"

    for cmd in base_commands:
        cmd_name = Path(cmd).name
        if cmd_name not in ALLOWED_COMMANDS:
            return False, f"Command not in allowlist: {cmd_name}"

    return True, ""


def validate_paths_in_command(command: str) -> tuple[bool, str]:
    """Check that any absolute paths in the command are within the sandbox."""
    if not ALLOWED_DIRECTORIES:
        return False, "No directories have been granted access"

    path_pattern = r'(?:^|\s)(/[^\s]+)'
    matches = re.findall(path_pattern, command)

    for path_str in matches:
        try:
            path = Path(path_str).resolve()
            in_sandbox = False
            for allowed in ALLOWED_DIRECTORIES:
                try:
                    if path.is_relative_to(allowed):
                        in_sandbox = True
                        break
                except ValueError:
                    continue
            if not in_sandbox:
                return False, f"Path outside sandbox: {path_str}"
        except (OSError, ValueError):
            continue

    return True, ""


# =============================================================================
# WINDOWS COMMAND TRANSLATION
# =============================================================================

# Maps Unix command patterns to Windows cmd.exe / PowerShell equivalents.
# Each entry is (unix_regex, windows_replacement).  Applied in order; the
# first match wins for that segment.  Replacements may use back-references.
_WIN_TRANSLATIONS: list[tuple[re.Pattern[str], str]] = [
    # ls [-flags] [path]  →  dir [path] (hide "Volume" header noise)
    (re.compile(r'^ls(\s+-[a-zA-Z]+)*(\s+.*)?$'), lambda m: (
        "dir /b" + (m.group(2) or "")
        if not (m.group(1) or "").strip()
        else "dir" + (m.group(2) or "")
    )),
    # cat file  →  type file
    (re.compile(r'^cat\s+(.+)$'), r'type \1'),
    # cp src dst  →  copy src dst
    (re.compile(r'^cp(\s+-r)?\s+(.+)\s+(\S+)$'), r'xcopy /E /I \2 \3'),
    # mv src dst  →  move src dst
    (re.compile(r'^mv\s+(.+)\s+(\S+)$'), r'move \1 \2'),
    # rm [-rf] path  →  del or rmdir
    (re.compile(r'^rm\s+(-rf?\s+|-r\s+|-f\s+)?(.+)$'), lambda m: (
        f'rmdir /S /Q {m.group(2)}' if (m.group(1) or '').strip() in ('-r', '-rf', '-fr')
        else f'del /F /Q {m.group(2)}'
    )),
    # mkdir [-p] dir  →  mkdir dir  (mkdir on Windows creates parents automatically)
    (re.compile(r'^mkdir\s+(-p\s+)?(.+)$'), r'mkdir \2'),
    # touch file  →  type nul >> file  (create empty file or update timestamp)
    (re.compile(r'^touch\s+(.+)$'), r'type nul >> \1'),
    # find . -name "pat" / find . -name 'pat' / find . -name pat  →  dir /S /B pat
    # The pattern strips optional surrounding single or double quotes from the name.
    (re.compile(r"""^find\s+\S+\s+-name\s+['"]*([^'"]+)['"]*$"""), r'dir /S /B \1'),
    # grep [-r] [-i] [-n] pattern [path]  →  findstr /S pattern [path]
    (re.compile(r'^grep\s+(-r\s+)?(-i\s+)?(-n\s+)?(.+)$'), lambda m: (
        'findstr /S'
        + (' /I' if m.group(2) else '')
        + (' /N' if m.group(3) else '')
        + f' {m.group(4)}'
    )),
    # head [-n N] file  →  PowerShell Get-Content ... -TotalCount N
    (re.compile(r'^head\s+(?:-n\s+(\d+)\s+)?(.+)$'), lambda m: (
        f'powershell -Command "Get-Content {m.group(2)} -TotalCount {m.group(1) or 10}"'
    )),
    # tail [-n N] file  →  PowerShell Get-Content ... -Tail N
    (re.compile(r'^tail\s+(?:-n\s+(\d+)\s+)?(.+)$'), lambda m: (
        f'powershell -Command "Get-Content {m.group(2)} -Tail {m.group(1) or 10}"'
    )),
    # wc -l file  →  PowerShell (Get-Content ...).Count
    (re.compile(r'^wc\s+-l\s+(.+)$'),
     r'powershell -Command "(Get-Content \1 | Measure-Object -Line).Lines"'),
    # sort file  →  sort file  (sort exists on Windows too)
    # diff file1 file2  →  fc file1 file2
    (re.compile(r'^diff\s+(.+)\s+(\S+)$'), r'fc \1 \2'),
    # echo — works on both; no translation needed
    # python / git / tar / zip — work natively; no translation needed
]


def _translate_segment(segment: str) -> str:
    """Translate a single shell command segment to its Windows equivalent."""
    s = segment.strip()
    for pattern, repl in _WIN_TRANSLATIONS:
        if callable(repl):
            m = pattern.match(s)
            if m:
                return repl(m)
        else:
            translated = pattern.sub(repl, s)
            if translated != s:
                return translated
    return s


# Regex that splits a command on shell operators while preserving the operators
_OPERATOR_SPLIT = re.compile(r'(\s*(?:\|\|?|&&|;)\s*)')


def translate_for_windows(command: str) -> str:
    """Translate a Unix shell command string to its Windows cmd.exe equivalent.

    Splits on operators (|, ||, &&, ;) and translates each segment individually,
    then re-joins with the original operators.
    """
    parts = _OPERATOR_SPLIT.split(command)
    out: list[str] = []
    for part in parts:
        if _OPERATOR_SPLIT.fullmatch(part):
            out.append(part)  # keep the operator as-is
        else:
            out.append(_translate_segment(part))
    return "".join(out)


# =============================================================================
# HEREDOC HANDLING (cross-platform)
# =============================================================================

# Matches:  cat > filename << 'MARKER'\n...content...\nMARKER
# The content between the markers is written to the file using Python,
# so this works on Windows cmd.exe and Unix shells alike.
_HEREDOC_RE = re.compile(
    r"""^cat\s+>\s*(\S+)\s+<<\s*['"]*(\w+)['"]*\s*\n(.*?)\n\2\s*$""",
    re.DOTALL,
)


def _try_heredoc(command: str, cwd: Path) -> tuple[bool, str]:
    """If command is a heredoc file-write, execute it via Python and return (True, result).

    Returns (False, "") if the command is not a heredoc so the caller can
    fall through to normal shell execution.
    """
    m = _HEREDOC_RE.match(command.strip())
    if not m:
        return False, ""

    filename = m.group(1).strip("'\"")
    content = m.group(3)

    # Resolve path and verify it stays inside the sandbox
    try:
        filepath = (cwd / filename).resolve()
    except Exception as e:
        return True, f"Error resolving path: {e}"

    in_sandbox = any(
        filepath == allowed or str(filepath).startswith(str(allowed) + os.sep)
        for allowed in ALLOWED_DIRECTORIES
    )
    if not in_sandbox:
        return True, f"Blocked: file path outside sandbox: {filepath}"

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        return True, f"Written {filepath.name} ({lines} lines)"
    except OSError as e:
        return True, f"Error writing file: {e}"


# =============================================================================
# SHELL TOOL
# =============================================================================

@tool
def run_shell(command: str) -> str:
    """Execute a shell command within the sandboxed directory.

    Allowed commands: ls, cat, cp, mv, rm, mkdir, find, grep, sort, sed, awk,
    python, bash, git, tar, zip, jq, and more standard Unix tools.
    Supports pipes (|) and chaining (&&, ||).

    Examples:
    - ls -la
    - find . -name "*.pdf" | head -10
    - mkdir -p images && mv *.jpg images/
    - python script.py
    - git status

    Args:
        command: Shell command to execute

    Returns:
        Command output (stdout + stderr)
    """
    log_tool_call("run_shell", {"command": command})

    if not ALLOWED_DIRECTORIES:
        return "Error: No directories have been granted access. Use 'grant' command first."

    is_valid, error = validate_command(command)
    if not is_valid:
        return f"Blocked: {error}\n\nAllowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}"

    is_valid, error = validate_paths_in_command(command)
    if not is_valid:
        return f"Blocked: {error}"

    # Handle heredoc (cat > file << 'EOF'...) natively — works on any OS
    handled, heredoc_result = _try_heredoc(command, ALLOWED_DIRECTORIES[0])
    if handled:
        return heredoc_result

    exec_command = translate_for_windows(command) if IS_WINDOWS else command

    try:
        result = subprocess.run(
            exec_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(ALLOWED_DIRECTORIES[0]),
            env={
                **os.environ,
                "HOME": str(ALLOWED_DIRECTORIES[0]),
            },
        )

        output = result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"[stderr]: {result.stderr}"

        if not output.strip():
            if result.returncode == 0:
                return "Command completed successfully (no output)"
            else:
                return f"Command failed with exit code {result.returncode}"

        if len(output) > 10000:
            output = output[:10000] + f"\n\n... [truncated, {len(output)} chars total]"

        return output

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# FILE WRITE TOOL
# =============================================================================

@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file inside the sandboxed directory.

    Use this to create or overwrite any file — scripts, markdown, config, data, etc.
    Intermediate directories are created automatically.

    Prefer this over shell heredoc for any file with more than a few lines.

    Args:
        path: Relative path to the file (e.g. "report.md", "scripts/parse.py").
              Absolute paths are rejected.
        content: Full text content to write.

    Returns:
        Confirmation with the absolute path and line count, or an error message.
    """
    log_tool_call("write_file", {"path": path, "content": f"<{len(content)} chars>"})

    if not ALLOWED_DIRECTORIES:
        return "Error: No directories have been granted access. Use 'grant' command first."

    # Reject absolute paths
    if Path(path).is_absolute():
        return f"Error: path must be relative, got absolute path: {path}"

    cwd = ALLOWED_DIRECTORIES[0]
    try:
        filepath = (cwd / path).resolve()
    except Exception as e:
        return f"Error resolving path: {e}"

    # Sandbox check
    in_sandbox = filepath == cwd or str(filepath).startswith(str(cwd) + os.sep)
    if not in_sandbox:
        return f"Blocked: path resolves outside sandbox: {filepath}"

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        return f"Written: {filepath}  ({lines} lines, {len(content)} chars)"
    except OSError as e:
        return f"Error writing file: {e}"


# =============================================================================
# WEB TOOLS
# =============================================================================

@tool
def fetch_url(url: str) -> str:
    """Fetch and extract text content from a URL.

    Useful for reading documentation, articles, or any web content.

    Args:
        url: URL to fetch (must start with http:// or https://)

    Returns:
        Extracted text content from the page
    """
    log_tool_call("fetch_url", {"url": url})

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4"

    if not url.startswith(('http://', 'https://')):
        return "Error: URL must start with http:// or https://"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OpenCowork/1.0)"}
        response = httpx.get(url, timeout=15, follow_redirects=True, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > 8000:
            text = text[:8000] + "\n\n... [truncated]"

        return text

    except httpx.HTTPStatusError as e:
        return f"HTTP Error: {e.response.status_code}"
    except httpx.TimeoutException:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets.

    Args:
        query: Search query

    Returns:
        Formatted search results
    """
    log_tool_call("search_web", {"query": query})

    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"

    try:
        results = list(DDGS().text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. **{r['title']}**\n"
                f"   {r['href']}\n"
                f"   {r.get('body', '')[:200]}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Error: {e}"
