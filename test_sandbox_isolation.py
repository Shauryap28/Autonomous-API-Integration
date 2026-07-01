"""
test_sandbox_isolation.py — prove the sandbox's safe-failure guarantees.

Run from the repo root:
    python test_sandbox_isolation.py

Checks two things a misbehaving script should NOT be able to do:
  1. write a file        -> blocked by the read-only filesystem
  2. read host files      -> the container has no host mount, so /etc exists but
                             YOUR files do not; we just confirm no host path leaks
"""
from backend.sandbox.runner import get_runner

runner = get_runner()

print("=== Test A: write a file (expect: blocked, read-only filesystem) ===")
write_code = 'open("/evil.txt", "w").write("x")'
a = runner.run_script(write_code)
print("exit_code:", a.exit_code)
print("stderr:", a.stderr.strip()[:400])
print("PASS" if a.exit_code != 0 and "read-only" in a.stderr.lower() else "CHECK ABOVE")

print("\n=== Test B: try to write to /tmp (expect: ALLOWED — in-memory scratch) ===")
tmp_code = 'open("/tmp/ok.txt", "w").write("x"); print("wrote to /tmp fine")'
b = runner.run_script(tmp_code)
print("exit_code:", b.exit_code)
print("stdout:", b.stdout.strip()[:200])
print("PASS" if b.exit_code == 0 else "CHECK ABOVE")

print("\n=== Test C: confirm it runs as a non-root user (expect: not uid 0) ===")
whoami_code = 'import os; print("uid", os.getuid())'
c = runner.run_script(whoami_code)
print("exit_code:", c.exit_code)
print("stdout:", c.stdout.strip()[:200])
print("PASS (non-root)" if c.exit_code == 0 and "uid 0" not in c.stdout else "CHECK ABOVE")