"""
runner — pick the execution backend based on settings.USE_SANDBOX.

  USE_SANDBOX = True  -> Docker sandbox (isolated, default)
  USE_SANDBOX = False -> local subprocess (fallback if Docker is unavailable)

Both expose the SAME run_script(code) -> ExecutionResult, so callers don't care
which one they got. This is the seam that makes the swap a one-line choice.
"""
from backend.config import settings


def get_runner():
    if settings.USE_SANDBOX:
        from backend.sandbox import docker_runner
        return docker_runner
    from backend.sandbox import local_runner
    return local_runner