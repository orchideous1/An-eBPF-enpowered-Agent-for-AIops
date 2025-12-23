import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# -------- File tools --------

def list_directory(path: str, recursive: bool = False, include_hidden: bool = False, max_entries: int = 2000) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"ok": False, "error": f"path not found: {p}"}
    results: List[Dict[str, Any]] = []

    def add_entry(entry_path: Path):
        try:
            is_dir = entry_path.is_dir()
            name = entry_path.name
            if not include_hidden and name.startswith('.'):
                return
            results.append({
                "name": name,
                "path": str(entry_path),
                "type": "dir" if is_dir else "file",
                "size": entry_path.stat().st_size if not is_dir else None,
            })
        except Exception as e:
            results.append({"name": entry_path.name, "path": str(entry_path), "type": "unknown", "error": str(e)})

    if recursive and p.is_dir():
        for root, dirs, files in os.walk(p):
            for d in dirs:
                add_entry(Path(root) / d)
                if len(results) >= max_entries:
                    break
            for f in files:
                add_entry(Path(root) / f)
                if len(results) >= max_entries:
                    break
            if len(results) >= max_entries:
                break
    elif p.is_dir():
        for child in p.iterdir():
            add_entry(child)
            if len(results) >= max_entries:
                break
    else:
        add_entry(p)

    truncated = len(results) >= max_entries
    return {"ok": True, "entries": results, "truncated": truncated}

def read_file(path: str, offset: int = 1, limit: Optional[int] = None, encoding: str = "utf-8") -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"ok": False, "error": f"file not found: {p}"}
    if p.is_dir():
        return {"ok": False, "error": f"path is a directory: {p}"}
    if offset < 1:
        offset = 1
    try:
        with p.open("r", encoding=encoding, errors="replace") as f:
            if offset == 1 and limit is None:
                content = f.read()
            else:
                lines = f.readlines()
                start = offset - 1
                end = None if limit is None else start + max(0, limit)
                content = "".join(lines[start:end])
        return {"ok": True, "path": str(p), "content": content}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def write_file(path: str, content: str, mode: str = "overwrite", encoding: str = "utf-8", create_dirs: bool = False) -> Dict[str, Any]:
    p = Path(path).expanduser()
    try:
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "w" if mode == "overwrite" else "a"
        with p.open(write_mode, encoding=encoding) as f:
            f.write(content)
        return {"ok": True, "path": str(p), "bytes": len(content)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------- Command tools --------

def _merge_env(extra_env: Optional[Dict[str, str]]) -> Dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            if v is not None:
                env[str(k)] = str(v)
    return env

def exec_shell(command: str,
               shell: str = "bash",
               workdir: Optional[str] = None,
               timeout: int = 120,
               env: Optional[Dict[str, str]] = None,
               use_sudo: bool = False) -> Dict[str, Any]:
    """
    Execute shell commands.
    - shell: 'bash' (default), 'sh', 'zsh', 'cmd', 'powershell'
    - use_sudo: when True, uses env var SUDO_PASSWORD for sudo -S
    """
    workdir = workdir or os.getcwd()
    merged_env = _merge_env(env)
    sudo_pw = os.getenv("SUDO_PASSWORD")
    ran_with_sudo = False

    if use_sudo:
        if not sudo_pw:
            return {"ok": False, "error": "SUDO_PASSWORD not set in environment"}
        if command.strip().startswith("sudo "):
            # Avoid double-sudo, but ensure -S is present
            command = command.strip()
            if " -S " not in command:
                command = command.replace("sudo ", "sudo -S -p '' ", 1)
        else:
            command = f"sudo -S -p '' {command}"
        ran_with_sudo = True

    try:
        if shell.lower() in ("bash", "sh", "zsh"):
            sh = "bash" if shell.lower() == "bash" else ("zsh" if shell.lower() == "zsh" else "sh")
            proc = subprocess.run(
                [sh, "-lc", command],
                input=(sudo_pw + "\n") if ran_with_sudo else None,
                cwd=workdir,
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        elif shell.lower() == "cmd":
            proc = subprocess.run(
                ["cmd", "/c", command],
                cwd=workdir, env=merged_env, capture_output=True, text=True, timeout=timeout
            )
        elif shell.lower() in ("powershell", "pwsh"):
            pwsh = "pwsh" if shutil.which("pwsh") else "powershell"
            proc = subprocess.run(
                [pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
                cwd=workdir, env=merged_env, capture_output=True, text=True, timeout=timeout
            )
        else:
            return {"ok": False, "error": f"unsupported shell: {shell}"}

        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "shell": shell,
            "sudo": ran_with_sudo,
            "cwd": workdir
        }
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": f"timeout after {timeout}s", "stdout": e.stdout, "stderr": e.stderr}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def exec_python(code: str,
                args: Optional[List[str]] = None,
                workdir: Optional[str] = None,
                timeout: int = 120,
                env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Execute Python code using current interpreter.
    """
    workdir = workdir or os.getcwd()
    merged_env = _merge_env(env)
    py = sys.executable or "python"
    argv = [py, "-c", code]
    if args:
        argv += list(map(str, args))

    try:
        proc = subprocess.run(
            argv,
            cwd=workdir,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "interpreter": py,
            "cwd": workdir
        }
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": f"timeout after {timeout}s", "stdout": e.stdout, "stderr": e.stderr}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------- JSON Schemas for registration --------

SCHEMA_LIST_DIRECTORY: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to list"},
        "recursive": {"type": "boolean", "default": False},
        "include_hidden": {"type": "boolean", "default": False},
        "max_entries": {"type": "integer", "default": 2000, "minimum": 1}
    },
    "required": ["path"]
}

SCHEMA_READ_FILE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer", "default": 1, "minimum": 1},
        "limit": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}], "default": None},
        "encoding": {"type": "string", "default": "utf-8"}
    },
    "required": ["path"]
}

SCHEMA_WRITE_FILE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
        "encoding": {"type": "string", "default": "utf-8"},
        "create_dirs": {"type": "boolean", "default": False}
    },
    "required": ["path", "content"]
}

SCHEMA_EXEC_SHELL: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Command to run"},
        "shell": {"type": "string", "enum": ["bash", "sh", "zsh", "cmd", "powershell"], "default": "bash"},
        "workdir": {"type": "string"},
        "timeout": {"type": "integer", "default": 120, "minimum": 1},
        "env": {"type": "object", "additionalProperties": {"type": "string"}},
        "use_sudo": {"type": "boolean", "default": False}
    },
    "required": ["command"]
}

SCHEMA_EXEC_PYTHON: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python code to run via -c"},
        "args": {"type": "array", "items": {"type": "string"}},
        "workdir": {"type": "string"},
        "timeout": {"type": "integer", "default": 120, "minimum": 1},
        "env": {"type": "object", "additionalProperties": {"type": "string"}}
    },
    "required": ["code"]
}

DEFAULT_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_directory",
        "description": "列出目录内容，支持递归与隐藏文件过滤",
        "parameters": SCHEMA_LIST_DIRECTORY,
        "function": list_directory,
        "category": "file",
    },
    {
        "name": "read_file",
        "description": "读取文本文件内容，支持按行偏移与限制",
        "parameters": SCHEMA_READ_FILE,
        "function": read_file,
        "category": "file",
    },
    {
        "name": "write_file",
        "description": "写入文本文件，支持覆盖或追加并可自动创建目录",
        "parameters": SCHEMA_WRITE_FILE,
        "function": write_file,
        "category": "file",
    },
    {
        "name": "exec_shell",
        "description": "执行 Shell 命令（bash/sh/zsh/cmd/powershell），可选 sudo 与超时",
        "parameters": SCHEMA_EXEC_SHELL,
        "function": exec_shell,
        "category": "command",
    },
    {
        "name": "exec_python",
        "description": "使用当前解释器执行 Python 代码（python -c）",
        "parameters": SCHEMA_EXEC_PYTHON,
        "function": exec_python,
        "category": "command",
    },
]

def register_default_tools(agent: Any, include_categories: Optional[List[str]] = None) -> None:
    """
    Register default file and command tools into a BaseAgent-compatible instance.
    """
    cats = set(include_categories) if include_categories else None
    for t in DEFAULT_TOOLS:
        if cats and t["category"] not in cats:
            continue
        agent.register_tool(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
            function=t["function"],
            category=t["category"]
        )
