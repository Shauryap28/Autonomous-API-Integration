"""
docker_runner — execute a generated script INSIDE an isolated Docker container.

Same contract as local_runner.run_script(code) -> ExecutionResult, so main.py
swaps one import and nothing upstream changes. Safety: the code runs in a
throwaway container from the `aaie-sandbox` image with
  - no host filesystem mount     (generated code can't see your files)
  - a hard memory cap            (no eating your RAM)
  - a CPU cap                    (no pegging your processor)
  - a read-only root filesystem  (can't write anywhere except an in-memory /tmp)
  - a non-root user              (no root powers; baked into the image)
  - a hard timeout               (infinite loops get killed)
Network stays ON (the script must reach the API); egress is NOT restricted to one
host (a known, stated limit).

How the code gets in (cross-platform): we base64-encode the script and pass it as
an argument to a tiny decode-and-exec bootstrap. No stdin socket (which breaks on
Windows named pipes), and still NO file shared from the host.
"""
import base64

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

from backend.config import settings
from backend.sandbox.local_runner import ExecutionResult  # reuse the same result type

# Decode the base64 script from argv and exec it. Kept as one line to pass cleanly.
_BOOTSTRAP = "import base64,sys; exec(base64.b64decode(sys.argv[1]).decode())"


def run_script(code, timeout=None):
    timeout = timeout or settings.SANDBOX_TIMEOUT
    client = docker.from_env()
    payload = base64.b64encode(code.encode("utf-8")).decode("ascii")

    container = None
    try:
        container = client.containers.run(
            image=settings.SANDBOX_IMAGE,
            command=["python", "-c", _BOOTSTRAP, payload],
            detach=True,                         # run in background so we can wait + time out
            mem_limit=settings.SANDBOX_MEM_LIMIT,
            nano_cpus=int(settings.SANDBOX_CPUS * 1_000_000_000),
            read_only=settings.SANDBOX_READONLY,
            network_disabled=False,              # must reach the API
            tmpfs={"/tmp": ""},                  # in-memory scratch (rootfs is read-only)
        )

        try:
            result = container.wait(timeout=timeout)
            exit_code = result.get("StatusCode", -1)
            timed_out = False
        except Exception:                        # wait() timed out (or transport hiccup)
            try:
                container.kill()
            except APIError:
                pass
            exit_code, timed_out = -1, True

        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", "replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", "replace")
        if timed_out:
            stderr += f"\n[killed: timeout after {timeout}s in sandbox]"

        return ExecutionResult(exit_code, stdout, stderr)

    except ImageNotFound:
        return ExecutionResult(
            -1, "",
            f"[sandbox] image '{settings.SANDBOX_IMAGE}' not found. "
            "Build it: docker build -t aaie-sandbox ./sandbox_image",
        )
    except (ContainerError, APIError) as e:
        return ExecutionResult(-1, "", f"[sandbox] docker error: {e}")
    finally:
        if container is not None:
            try:
                container.remove(force=True)     # bulldoze the room
            except APIError:
                pass