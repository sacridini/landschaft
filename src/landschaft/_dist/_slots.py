# SPDX-License-Identifier: GPL-3.0-or-later
"""Child processes that run tasks (``_child``), kept alive between tasks so landschaft is
imported once per process. A task that is cancelled or crashes takes its process with it;
the next task starts a new one.

Tasks are sent as JSON over a pipe to ``python -m landschaft._dist._child``, never with
multiprocessing: that would re-run the user's script in every child on Windows.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

START_TIMEOUT = 300.0


def child_env(cores: int) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(max(1, cores))
    env["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "darwin":
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


class Slot:
    """One child process."""

    def __init__(self, cores: int, log: Path):
        self.log = log
        self.cores = cores
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()

    def _start(self) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with open(self.log, "ab") as err:
            self.proc = subprocess.Popen([sys.executable, "-m", "landschaft._dist._child"], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=err, cwd=str(self.log.parent),
                                         env=child_env(self.cores), creationflags=flags)
        self._lines = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self._lines), name="landschaft-dist-child",
                         daemon=True).start()
        reply = self._wait(None, START_TIMEOUT)
        if reply is None or not reply.get("ready"):
            self.kill()
            raise RuntimeError(f"the child process did not start: {self._tail()}")

    @staticmethod
    def _read(proc, lines) -> None:
        for raw in proc.stdout:
            try:
                lines.put(json.loads(raw))
            except ValueError:
                continue
        lines.put(None)

    def _wait(self, cancel: threading.Event | None, timeout: float | None = None):
        """The next reply; None if the process ended, "cancelled" if ``cancel`` was set."""
        start = time.monotonic()
        while True:
            try:
                return self._lines.get(timeout=0.25)
            except queue.Empty:
                pass
            if cancel is not None and cancel.is_set():
                return "cancelled"
            if timeout is not None and time.monotonic() - start > timeout:
                return None

    def _tail(self) -> str:
        try:
            lines = self.log.read_text(errors="replace").strip().splitlines()
        except OSError:
            return "no output"
        return "\n".join(lines[-8:]) or "no output"

    def run(self, op: str, params: dict, inputs: Path, out: Path, threads: int,
            cancel: threading.Event | None = None) -> tuple[bool, float, str, dict]:
        """(finished, seconds, error message, the operation's stats)."""
        start = time.monotonic()
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        out.mkdir(parents=True, exist_ok=True)
        request = {"op": op, "params": params, "inputs": str(inputs), "out": str(out), "threads": int(threads)}
        try:
            self.proc.stdin.write((json.dumps(request) + "\n").encode())
            self.proc.stdin.flush()
        except OSError:
            self.kill()
            return False, time.monotonic() - start, f"the child process died: {self._tail()}", {}
        try:
            reply = self._wait(cancel)
        except BaseException:      # Ctrl+C: never leave the task running
            self.kill()
            raise
        if reply == "cancelled":
            self.kill()
            return False, time.monotonic() - start, "cancelled", {}
        if reply is None:
            self.kill()
            return False, time.monotonic() - start, f"the child process crashed: {self._tail()}", {}
        if reply.get("ok"):
            return True, float(reply.get("seconds") or 0.0), "", reply.get("stats") or {}
        message = str(reply.get("traceback") or reply.get("error") or "failed")
        return False, float(reply.get("seconds") or 0.0), message, {}

    def kill(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.proc = None


class Slots:
    """A pool of child processes for a machine with ``cores`` cores."""

    def __init__(self, cores: int, folder: Path):
        self.cores, self.folder = cores, Path(folder)
        self._idle: list[Slot] = []
        self._all: list[Slot] = []
        self._lock = threading.Lock()

    def acquire(self) -> Slot:
        with self._lock:
            if self._idle:
                return self._idle.pop()
            slot = Slot(self.cores, self.folder / f"child-{len(self._all) + 1}.log")
            self._all.append(slot)
            return slot

    def release(self, slot: Slot) -> None:
        with self._lock:
            self._idle.append(slot)

    def close(self) -> None:
        with self._lock:
            slots, self._all, self._idle = list(self._all), [], []
        for slot in slots:
            if slot.proc is not None:
                try:
                    slot.proc.stdin.close()
                    slot.proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            slot.kill()
