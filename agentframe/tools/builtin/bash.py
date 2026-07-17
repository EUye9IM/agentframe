from __future__ import annotations

import subprocess

from agentframe import function_tool


@function_tool(name="bash", description="Execute a bash command. Returns stdout+stderr. Use for file ops, git, grep, ls, etc.")
def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=None,
        )
        output = result.stdout.strip()
        if result.stderr:
            stderr = result.stderr.strip()
            if stderr:
                output = (output + "\n" + stderr).strip()
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"
    except Exception as e:
        return f"Error: {e}"
