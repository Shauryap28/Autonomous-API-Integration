# Minimal sandbox image for running untrusted, LLM-generated fetch scripts.
# Intentionally bare: Python + requests only, run as a non-root user.

FROM python:3.11-slim

# Only the one library a fetch script needs. No database driver, no extras —
# least privilege: if the tool isn't in the room, bad code can't use it.
RUN pip install --no-cache-dir requests

# Run as an unprivileged user, not root (defense in depth: even if something
# goes wrong inside, the process has no root powers in the container).
RUN useradd --create-home sandboxuser
USER sandboxuser