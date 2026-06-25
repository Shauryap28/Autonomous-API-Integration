"""
local_runner — execute a generated Python script and capture its output.

Phase 2: runs the script LOCALLY and UNSANDBOXED (directly in this venv), piped
via stdin. This is acceptable only for one controlled test against a read-only
public API — it is exactly the risk Phase 3's Docker sandbox removes.

The signature (code in -> exit_code/stdout/stderr out) is the seam Phase 3 swaps
for Docker WITHOUT changing anything upstream.
"""
import subprocess
import sys
from dataclasses import dataclass

from backend.config import settings


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str


def run_script(code, timeout=None):
    timeout = timeout or settings.EXEC_TIMEOUT
    try:
        proc = subprocess.run(
            [sys.executable, "-"],     # read the program from stdin
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(-1, e.stdout or "", (e.stderr or "") + f"\n[killed: timeout after {timeout}s]")