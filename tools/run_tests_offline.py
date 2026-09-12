"""Run repository tests with network access disabled after tokenizer setup.

Install requirements and cache cl100k_base before invoking this command. The
local model, provider APIs and dataset downloads are never test prerequisites.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def offline():
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is forbidden during the offline test suite")

    names = ((socket, "create_connection"), (socket, "getaddrinfo"),
             (socket.socket, "connect"), (socket.socket, "connect_ex"),
             (socket.socket, "sendto"))
    saved = [(owner, name, getattr(owner, name)) for owner, name in names]
    try:
        for owner, name in names:
            setattr(owner, name, blocked)
        yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


def main() -> int:
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    (ROOT / ".runs").mkdir(exist_ok=True)
    with offline():
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
