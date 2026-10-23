"""One run at a time. The operating system holds the lock, not the file's existence, so
it is released when the process ends however it ends: a crash or power cut can't leave
a stale lock behind."""
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from newsagg.config import ROOT

LOCK = ROOT / "data" / "run.lock"


class AlreadyRunning(Exception):
    pass


@contextmanager
def run_lock(path: Path = LOCK) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        try:
            if sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)   # NB: fail at once, don't wait
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise AlreadyRunning(str(path)) from None
        try:
            yield
        finally:
            if sys.platform == "win32":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
