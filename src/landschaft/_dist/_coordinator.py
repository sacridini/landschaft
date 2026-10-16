# SPDX-License-Identifier: GPL-3.0-or-later
"""The coordinator: hands the tasks of a job out to this computer and to the helpers that
connect, and gives each finished task's outputs to the job.

Helpers pull: each asks for a task when it has room (its tasks running, one being
fetched), so fast and slow machines balance by themselves. The heaviest tasks go first.
Near the end, an idle machine that would finish a straggler sooner runs it too, and the
first result wins. The coordinator lives as long as its ``Helpers``: helpers stay
connected between jobs.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import _group, _net, _system, _tasks
from ._slots import Slots
from ._tasks import Job, Task, format_bytes, format_seconds

MAX_FAILURES = 2          # a task that fails this many times is given up
MAX_TRANSFER_ERRORS = 5   # transfers of a task that broke: back to the queue, then counted as failures
HEARTBEAT_TIMEOUT = 60.0  # a helper silent this long is gone; its tasks go back to the queue
LOST_AFTER = 20.0         # an attempt a helper does not report holding is lost
DISK_CHUNK = 16 << 20     # files are read one chunk at a time: sequential on a hard disk
NO_HELPERS_HINT = 20.0
EMA = 0.5


@dataclass
class Machine:
    name: str
    local: bool
    cores: int
    address: str = ""
    token: str = ""
    version: str = ""
    last_seen: float = field(default_factory=time.monotonic)
    gone: bool = False
    done: int = 0
    compute_rate: float | None = None    # task weight per second of processing
    fetch_rate: float | None = None      # task weight per second of fetching
    compute_s: float = 0.0
    starved_s: float = 0.0
    to_cancel: set = field(default_factory=set)
    via: dict = field(default_factory=dict)

    def estimate(self, weight: float) -> float | None:
        if self.compute_rate is None or (not self.local and self.fetch_rate is None):
            return None
        return (0.0 if self.local else weight / self.fetch_rate) + weight / self.compute_rate


@dataclass
class _Task:
    task: Task
    state: str = "pending"      # pending, running, collecting (its outputs being taken), done, failed
    attempts: list = field(default_factory=list)
    failures: int = 0
    transfer_errors: int = 0
    failed_on: set = field(default_factory=set)
    errors: list = field(default_factory=list)


@dataclass
class _Attempt:
    id: str
    task: _Task
    machine: Machine
    started: float
    cancel: threading.Event = field(default_factory=threading.Event)
    closed: bool = False


@dataclass
class JobResult:
    """What a job did: the tasks finished, the failed ones (name -> error) and how many
    tasks each computer ran."""

    done: list
    failed: dict
    machines: dict
    seconds: float

    @property
    def ok(self) -> bool:
        return not self.failed


class _JobState:
    def __init__(self, job: Job, workdir: Path):
        self.job = job
        self.tasks = [_Task(t) for t in sorted(job.tasks, key=lambda t: t.weight, reverse=True)]
        self.workdir = workdir
        self.finished = threading.Event()
        self.started = time.monotonic()
        self.results: dict | None = None      # where helpers that see the results folder put outputs
        self.counted: dict = {}
        if not self.tasks:
            self.finished.set()


class _Printer:
    def __init__(self, out: Callable[[str], None] | None, verbose: bool):
        self._out = out
        self._verbose = verbose
        self._tty = out is None and verbose and sys.stdout.isatty()
        self._status = ""
        self._lock = threading.Lock()

    def line(self, text: str) -> None:
        if not self._verbose:
            return
        with self._lock:
            if self._out is not None:
                self._out(text)
                return
            if self._tty and self._status:
                sys.stdout.write("\r" + " " * len(self._status) + "\r")
            print(text, flush=True)
            if self._tty and self._status:
                sys.stdout.write(self._status)
                sys.stdout.flush()

    def status(self, text: str) -> None:
        if not self._verbose:
            return
        with self._lock:
            if self._tty:
                pad = max(0, len(self._status) - len(text))
                sys.stdout.write("\r" + text + " " * pad)
                sys.stdout.flush()
                self._status = text
            elif self._out is None:
                print(text, flush=True)
            else:
                self._out(text)

    def end(self) -> None:
        with self._lock:
            if self._tty and self._status:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._status = ""

    @property
    def interval(self) -> float:
        return 1.0 if self._tty else 30.0


class _Limited:
    """A request body as a file of exactly Content-Length bytes."""

    def __init__(self, stream, length: int):
        self._stream, self._left = stream, length

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        n = self._left if n is None or n < 0 else min(n, self._left)
        data = self._stream.read(n)
        self._left -= len(data)
        return data


class Coordinator:
    def __init__(self, *, local: bool = True, local_threads: int | None = None, helpers: bool = True,
                 port: int = _group.TCP_PORT, discovery: bool = True, out: Callable[[str], None] | None = None,
                 host: str = "", verbose: bool = True):
        from .._threads import resolve

        if not local and not helpers:
            raise ValueError("nothing would run the tasks: use this computer (local) or helpers, or both")
        self.helpers, self.discovery, self.port, self.host = helpers, discovery, port, host
        self.version = _group.version()
        self.name = _group.machine_name()
        self._print = _Printer(out, verbose)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._disk = threading.Lock()
        self._state: _JobState | None = None
        self._jobs = 0
        self._attempts: dict[str, _Attempt] = {}
        self._machines: list[Machine] = []
        self._by_token: dict[str, Machine] = {}
        self._counter = 0
        self._sent = deque()             # (time, bytes) of input data sent, for the rate
        self._warned: set = set()
        self._started = time.monotonic()
        self._threads: list[threading.Thread] = []
        self._server = self._responder = None
        self._tmp = tempfile.TemporaryDirectory(prefix="landschaft-dist-")
        self._started_up = False
        self.local_machine: Machine | None = None
        self._slots: Slots | None = None
        if local:
            cores = resolve(local_threads)
            self.local_machine = Machine(f"{self.name} (this computer)", True, cores)
            self._machines.append(self.local_machine)
            self._slots = Slots(cores, Path(self._tmp.name) / "local")

    # -------------------------------------------------------------- lifecycle

    def start(self) -> "Coordinator":
        if self._started_up:
            return self
        self._started_up = True
        if self.helpers:
            secret = _group.ensure_secret()     # a group of one until others join (landschaft helpers invite)
            self._gid = _group.group_id(secret)
            cert, key, fingerprint = _group.certificate(Path(self._tmp.name) / "tls")
            self._handshakes = _net.Handshakes(secret, fingerprint)
            self._server = _net.Server(self, self.port, cert, key, host=self.host).start()
            self.port = self._server.port
            if self.discovery:
                try:
                    self._responder = _net.Responder(self._answer).start()
                except OSError as e:
                    self._print.line(f"Discovery is off (UDP port {_group.UDP_PORT}: {e}); helpers can still "
                                     f"connect with: landschaft worker --connect {self._address()}:{self.port}")
        if self.local_machine is not None:
            self._spawn(self._local_loop, "landschaft-dist-local")
        self._spawn(self._monitor, "landschaft-dist-monitor")
        return self

    def _spawn(self, target, name) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def run(self, job: Job) -> JobResult:
        """Run every task of ``job`` (one job at a time) and wait for them."""
        self.start()
        with self._lock:
            if self._state is not None and not self._state.finished.is_set():
                raise RuntimeError("one job at a time per Helpers")
            self._jobs += 1
            job.id = self._jobs
            for i, t in enumerate(job.tasks):
                t.id = i
            if job.results_dir is not None:
                workdir = Path(job.results_dir) / _tasks.PARTIAL / f"job-{secrets.token_hex(4)}"
            else:
                workdir = Path(self._tmp.name) / f"job-{job.id}"
            workdir.mkdir(parents=True, exist_ok=True)
            state = _JobState(job, workdir)
            if self.helpers and job.results_dir is not None:
                state.results = self._results_probe(workdir)
            self._state = state
            state.counted = {m.name: m.done for m in self._machines}
        self._wake.set()
        try:
            with _system.keep_awake():
                while not state.finished.wait(0.5):
                    pass
        except KeyboardInterrupt:
            self._print.end()
            self._print.line("Stopped.")
            self._abort(state)
            raise
        self._print.end()
        result = self._result(state)
        shutil.rmtree(workdir, ignore_errors=True)
        if job.results_dir is not None:
            try:
                (Path(job.results_dir) / _tasks.PARTIAL).rmdir()
            except OSError:
                pass
        return result

    def _abort(self, state: _JobState) -> None:
        with self._lock:
            for a in list(self._attempts.values()):
                if a.task in state.tasks:
                    self._close(a)
                    if not a.machine.local:
                        a.machine.to_cancel.add(a.id)
            for t in state.tasks:
                if t.state in ("pending", "running", "collecting"):
                    t.state = "failed"
                    t.errors.append("stopped")
            state.finished.set()
        shutil.rmtree(state.workdir, ignore_errors=True)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            for attempt in self._attempts.values():
                attempt.cancel.set()
            if self._state is not None:
                self._state.finished.set()
        time.sleep(0.5 if self._server is not None else 0)   # helpers asking now hear "done"
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(timeout=10)
        if self._responder is not None:
            self._responder.close()
            self._responder = None
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._slots is not None:
            self._slots.close()
        self._tmp.cleanup()
        self._print.end()

    @property
    def machines(self) -> list:
        with self._lock:
            return [m for m in self._machines if not m.gone]

    def _address(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.0.2.1", 9))     # no packet is sent; picks the outgoing interface
                return s.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())

    def _answer(self, message: dict, addr) -> dict | None:
        if message.get("kind") == "helper" and message.get("group") == self._gid and not self._stop.is_set():
            return {"kind": "coordinator", "group": self._gid, "port": self.port, "name": self.name}
        return None

    # -------------------------------------------------------------- scheduling

    def _new_attempt(self, task: _Task, machine: Machine) -> _Attempt:
        self._counter += 1
        attempt = _Attempt(f"{self._state.job.id}.{task.task.id}.{self._counter}", task, machine, time.monotonic())
        task.attempts.append(attempt)
        task.state = "running"
        self._attempts[attempt.id] = attempt
        return attempt

    def _next(self, machine: Machine) -> tuple:
        """("task", attempt), ("wait", seconds) or ("done", None)."""
        with self._lock:
            if self._stop.is_set():
                return "done", None
            state = self._state
            if state is None or state.finished.is_set():
                return "wait", 2.0
            pending = [t for t in state.tasks if t.state == "pending"]
            if pending:
                task = ([t for t in pending if machine.name not in t.failed_on] or pending)[0]
                if not machine.local and self._local_sooner(task, machine, len(pending)):
                    return "wait", 2.0
                return "task", self._new_attempt(task, machine)
            task = self._straggler(machine)
            if task is not None:
                return "task", self._new_attempt(task, machine)
            return "wait", 3.0

    def _remaining(self, machine: Machine, now: float) -> float:
        """Estimated seconds until ``machine`` is free (0 when idle or unknown)."""
        left = 0.0
        for a in self._attempts.values():
            if a.machine is machine and not a.closed:
                est = machine.estimate(a.task.task.weight)
                if est is not None:
                    left = max(left, a.started + est - now)
        return left

    def _local_sooner(self, task: _Task, machine: Machine, n_pending: int) -> bool:
        """The last task stays here when this computer would finish it before the helper."""
        local = self.local_machine
        if local is None or n_pending > 1 or local.compute_rate is None:
            return False
        estimate = machine.estimate(task.task.weight)
        if estimate is None:
            return False
        now = time.monotonic()
        theirs = self._remaining(machine, now) + estimate    # a helper asking ahead is still busy
        ours = self._remaining(local, now) + local.estimate(task.task.weight)
        return ours < theirs

    def _straggler(self, machine: Machine) -> _Task | None:
        """A running task that ``machine`` would finish clearly sooner than its holder."""
        now = time.monotonic()
        if self._remaining(machine, now) > 0:
            return None
        best, best_finish = None, None
        for task in self._state.tasks:
            if task.state != "running" or any(a.machine is machine for a in task.attempts):
                continue
            active = [a for a in task.attempts if not a.closed]
            if len(active) != 1:
                continue
            holder = active[0]
            est_holder = holder.machine.estimate(task.task.weight)
            est_mine = machine.estimate(task.task.weight)
            if est_holder is None or est_mine is None:
                continue
            finish = holder.started + est_holder
            margin = max(30.0, 0.1 * (finish - now))
            if now + est_mine < finish - margin and (best_finish is None or finish > best_finish):
                best, best_finish = task, finish
        return best

    def _close(self, attempt: _Attempt) -> None:
        attempt.closed = True
        attempt.cancel.set()
        self._attempts.pop(attempt.id, None)

    def _owning_state(self, attempt: _Attempt) -> _JobState | None:
        state = self._state
        return state if state is not None and attempt.task in state.tasks else None

    def _complete(self, attempt: _Attempt, folder: Path, stats: dict) -> bool:
        with self._lock:
            task, machine = attempt.task, attempt.machine
            state = self._owning_state(attempt)
            if attempt.closed or state is None or task.state in ("collecting", "done", "failed"):
                self._close(attempt)
                return False
            self._close(attempt)
            task.state = "collecting"
            for other in task.attempts:
                if not other.closed:
                    self._close(other)
                    if not other.machine.local:
                        other.machine.to_cancel.add(other.id)
            weight = task.task.weight
            compute_s = max(float(stats.get("compute_s") or 0), 1e-3)
            machine.compute_rate = _ema(machine.compute_rate, weight / compute_s)
            fetch_s = float(stats.get("fetch_s") or 0)
            if not machine.local and fetch_s > 0.05:
                machine.fetch_rate = _ema(machine.fetch_rate, weight / fetch_s)
            elif not machine.local and machine.fetch_rate is None:
                machine.fetch_rate = float("inf")
            machine.done += 1
            machine.compute_s += compute_s
            machine.starved_s += float(stats.get("starved_s") or 0)
            via = str(stats.get("via") or "local")
            machine.via[via] = machine.via.get(via, 0) + 1
        try:
            if state.job.collect is not None:
                state.job.collect(task.task, folder, {**stats, "machine": machine.name})
            outcome = "done"
        except Exception as e:  # noqa: BLE001  (the outputs could not be taken: the task failed)
            outcome = "failed"
            task.errors.append(f"{machine.name}: the outputs could not be read: {type(e).__name__}: {e}")
        with self._lock:
            task.state = outcome
            self._check_finished(state)
        self._wake.set()
        return True

    def _fail(self, attempt: _Attempt, message: str) -> None:
        with self._lock:
            task, machine = attempt.task, attempt.machine
            if attempt.closed:
                return
            self._close(attempt)
            task.failures += 1
            task.failed_on.add(machine.name)
            task.errors.append(f"{machine.name}: {message}")
            if task.state in ("collecting", "done", "failed") or any(not a.closed for a in task.attempts):
                return
            task.state = "failed" if task.failures >= MAX_FAILURES else "pending"
            lines = message.strip().splitlines()
            first = lines[-1] if lines else "no message"
            again = "given up" if task.state == "failed" else "will try again"
            self._print.line(f"! {task.task.name} failed on {machine.name} ({again}): {first}")
            state = self._owning_state(attempt)
            if state is not None:
                self._check_finished(state)
        self._wake.set()

    def _lose(self, attempt: _Attempt, transfer_error: str | None = None) -> None:
        """An attempt that will never report (its helper left, or a transfer broke): the task
        goes back to the queue, without counting as a failure, unless its transfers keep
        breaking."""
        with self._lock:
            if attempt.closed:
                return
            if transfer_error is not None:
                attempt.task.transfer_errors += 1
                if attempt.task.transfer_errors > MAX_TRANSFER_ERRORS:
                    self._fail(attempt, transfer_error)
                    return
            self._close(attempt)
            task = attempt.task
            if task.state == "running" and not any(not a.closed for a in task.attempts):
                task.state = "pending"
        self._wake.set()

    def _check_finished(self, state: _JobState) -> None:
        if all(t.state in ("done", "failed") for t in state.tasks):
            state.finished.set()
            self._wake.set()

    # -------------------------------------------------------------- this computer

    def _local_loop(self) -> None:
        machine = self.local_machine
        used = [0]
        cond = threading.Condition()

        def run_one(attempt: _Attempt, threads: int) -> None:
            try:
                self._run_local(attempt, threads)
            finally:
                with cond:
                    used[0] -= threads
                    cond.notify_all()
                self._wake.set()

        while not self._stop.is_set():
            state = self._state
            if state is None or state.finished.is_set():
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            threads = state.job.threads_on(machine.cores)
            with cond:
                if used[0] and used[0] + threads > machine.cores:
                    cond.wait(0.5)
                    continue
            kind, attempt = self._next(machine)
            if kind != "task":
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            with cond:
                used[0] += threads
            threading.Thread(target=run_one, args=(attempt, threads), name="landschaft-dist-task",
                             daemon=True).start()

    def _run_local(self, attempt: _Attempt, threads: int) -> None:
        task = attempt.task.task
        state = self._owning_state(attempt)
        if state is None:
            return
        folder = state.workdir / f"{_safe(task.name)}-{attempt.id}"
        out = folder / "out"
        slot = self._slots.acquire()
        try:
            ok, seconds, message, stats = slot.run(task.op, task.params, Path(task.root), out, threads,
                                                   attempt.cancel)
        except Exception as e:  # noqa: BLE001
            ok, seconds, message, stats = False, 0.0, f"{type(e).__name__}: {e}", {}
        finally:
            self._slots.release(slot)
        if ok:
            self._complete(attempt, out, {**stats, "via": "local", "compute_s": seconds})
        elif attempt.cancel.is_set():
            self._lose(attempt)
        else:
            self._fail(attempt, message)
        shutil.rmtree(folder, ignore_errors=True)

    # -------------------------------------------------------------- the helpers' requests

    def handle(self, handler, method: str, path: str) -> None:
        route = path.split("?", 1)[0].rstrip("/")
        if method == "POST" and route == "/hello":
            handler.send_json(200, self._handshakes.hello(handler.read_json()))
            return
        if method == "POST" and route == "/auth":
            self._auth(handler)
            return
        machine = self._by_token.get(handler.headers.get("X-Landschaft-Session") or "")
        if machine is None or machine.gone:
            handler.send_json(401, {"error": "not authenticated"})
            return
        machine.last_seen = time.monotonic()
        parts = route.strip("/").split("/")
        if method == "POST" and route == "/next":
            handler.read_body()
            self._serve_next(handler, machine)
        elif method == "GET" and parts[0] == "file" and len(parts) == 3:
            self._serve_file(handler, machine, parts[1], parts[2])
        elif method == "POST" and route == "/heartbeat":
            self._heartbeat(handler, machine)
        elif method == "PUT" and parts[0] == "result" and len(parts) == 2:
            self._receive(handler, machine, parts[1])
        elif method == "POST" and parts[0] == "fail" and len(parts) == 2:
            body = handler.read_json()
            attempt = self._attempts.get(parts[1])
            if attempt is not None and attempt.machine is machine:
                message = str(body.get("error") or "failed")[:4000]
                if body.get("transfer"):
                    self._lose(attempt, transfer_error=message)
                else:
                    self._fail(attempt, message)
            handler.send_json(200, {})
        elif method == "POST" and route == "/bye":
            handler.read_body()
            self._leave(machine, "left")
            handler.send_json(200, {})
        else:
            handler.send_json(404, {"error": f"no {method} {route}"})

    def _auth(self, handler) -> None:
        body = handler.read_json()
        if not self._handshakes.auth(body):
            handler.send_json(403, {"error": "not in this group"})
            return
        info = body.get("info") if isinstance(body.get("info"), dict) else {}
        version = str(info.get("version", "?"))
        name = "".join(c for c in str(info.get("name") or "helper") if c.isprintable())[:64] or "helper"
        if version != self.version:
            self._warn_once(f"version:{name}",
                            f"! {name} has landschaft {version}; this computer has {self.version}. It will not get "
                            f"tasks until both have the same version (pip install -U landschaft=={self.version}).")
            handler.send_json(409, {"error": f"this computer has landschaft {version}; {self.name} uses "
                                             f"{self.version}. Update with: pip install landschaft=={self.version}",
                                    "version": self.version})
            return
        if str(info.get("code")) != _group.code_fingerprint():
            self._warn_once(f"code:{name}",
                            f"! {name} has landschaft {version} too, but with other code (a development install on "
                            "one of them?). It will not get tasks until both have the same code.")
            handler.send_json(409, {"error": f"this computer has landschaft {version} with other code than "
                                             f"{self.name} (a development install on one of them?): install the same "
                                             "landschaft on both"})
            return
        address = handler.client_address[0]
        with self._lock:
            for old in self._machines:
                if not old.local and not old.gone and old.name == name and old.address == address:
                    self._leave(old, "reconnected")     # a helper that restarted
            taken = {m.name for m in self._machines if not m.gone}
            unique, n = name, 2
            while unique in taken:
                unique, n = f"{name}#{n}", n + 1
            try:
                cores = max(1, int(info.get("cores") or 1))
            except (TypeError, ValueError):
                cores = 1
            machine = Machine(unique, False, cores, address=address, token=secrets.token_hex(32), version=version)
            self._machines.append(machine)
            self._by_token[machine.token] = machine
        self._print.line(f"+ {machine.name} ({address}, {cores} cores) joined")
        handler.send_json(200, {"token": machine.token, "name": self.name})

    def _serve_next(self, handler, machine: Machine) -> None:
        kind, value = self._next(machine)
        if kind == "done":
            handler.send_json(200, {"done": True})
        elif kind == "wait":
            handler.send_json(200, {"wait": value})
        else:
            handler.send_json(200, {"task": self._payload(value)})

    def _payload(self, attempt: _Attempt) -> dict:
        task = attempt.task.task
        state = self._state
        files = [{"index": i, "name": f.name, "size": f.size, "key": f.key, "probe": f.probe, "path": f.shared_path}
                 for i, f in enumerate(task.files)]
        payload = {"attempt": attempt.id, "job": state.job.id, "label": state.job.label, "name": task.name,
                   "op": task.op, "params": task.params, "threads": state.job.threads_on(attempt.machine.cores),
                   "files": files}
        if state.results is not None:
            payload["results"] = {**state.results, "name": self._placed_name(attempt)}
        return payload

    def _placed_name(self, attempt: _Attempt) -> str:
        return f"{_safe(attempt.task.task.name)}-{attempt.id}-placed"

    def _results_probe(self, workdir: Path) -> dict | None:
        """A file in the job's folder by which a helper that sees the folder too (a NAS)
        recognizes it, and then writes its outputs there instead of sending them."""
        try:
            token = secrets.token_hex(16)
            name = f"probe-{token[:8]}.txt"
            (workdir / name).write_text(token)
        except OSError:
            return None
        return {"path": _tasks.network_path(workdir), "probe": name, "token": token}

    def _serve_file(self, handler, machine: Machine, attempt_id: str, index: str) -> None:
        attempt = self._attempts.get(attempt_id)
        if attempt is None or attempt.machine is not machine or attempt.closed:
            handler.send_json(404, {"error": "no such task"})
            return
        try:
            f = attempt.task.task.files[int(index)]
        except (ValueError, IndexError):
            handler.send_json(404, {"error": "no such file"})
            return
        size = os.path.getsize(f.path)

        def chunks():
            with open(f.path, "rb") as src:
                while True:
                    with self._disk:       # one read at a time: sequential on a hard disk
                        data = src.read(DISK_CHUNK)
                    if not data:
                        return
                    for i in range(0, len(data), _net.CHUNK):
                        piece = data[i:i + _net.CHUNK]
                        yield piece
                        self._count_sent(len(piece))

        handler.send_stream(size, chunks())

    def _heartbeat(self, handler, machine: Machine) -> None:
        body = handler.read_json()
        held = {str(a) for a in body.get("held", [])}
        now = time.monotonic()
        lost = []
        with self._lock:
            for attempt in list(self._attempts.values()):
                if attempt.machine is machine and attempt.id not in held and now - attempt.started > LOST_AFTER:
                    lost.append(attempt)
            cancel = sorted(machine.to_cancel & held)
            machine.to_cancel &= held
        for attempt in lost:
            self._lose(attempt)
        handler.send_json(200, {"cancel": cancel})

    def _receive(self, handler, machine: Machine, attempt_id: str) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        body = _Limited(handler.rfile, length)
        attempt = self._attempts.get(attempt_id)
        state = self._owning_state(attempt) if attempt is not None else None
        if attempt is None or state is None or attempt.machine is not machine or attempt.closed:
            while body.read(_net.CHUNK):      # drain: keep the connection usable
                pass
            handler.send_json(200, {"accepted": False})
            return
        try:
            stats = json.loads(handler.headers.get("X-Landschaft-Stats") or "{}")
        except ValueError:
            stats = {}
        stats = stats if isinstance(stats, dict) else {}
        placed = handler.headers.get("X-Landschaft-Placed")
        if placed is not None:
            while body.read(_net.CHUNK):
                pass
            folder = state.workdir / self._placed_name(attempt)
            if placed != self._placed_name(attempt) or not _tasks.check_placed(folder):
                shutil.rmtree(folder, ignore_errors=True)
                self._lose(attempt, transfer_error="the outputs placed in the results folder are not there")
                handler.send_json(200, {"accepted": False})
                return
            accepted = self._complete(attempt, folder, {**stats, "results": "shared"})
            shutil.rmtree(folder, ignore_errors=True)
            handler.send_json(200, {"accepted": accepted})
            return
        folder = state.workdir / f"{_safe(attempt.task.task.name)}-{attempt.id}"
        out = folder / "out"
        try:
            _tasks.unpack(body, out)
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(folder, ignore_errors=True)
            if isinstance(e, ValueError):      # refused contents
                while body.read(_net.CHUNK):
                    pass
                self._fail(attempt, f"bad outputs: {e}")
                handler.send_json(200, {"accepted": False})
            else:                              # the upload broke
                self._lose(attempt, transfer_error=f"the outputs did not arrive: {type(e).__name__}: {e}")
                handler.close_connection = True
            return
        accepted = self._complete(attempt, out, stats)
        shutil.rmtree(folder, ignore_errors=True)
        handler.send_json(200, {"accepted": accepted})

    def _leave(self, machine: Machine, why: str) -> None:
        with self._lock:
            if machine.gone:
                return
            machine.gone = True
            self._by_token.pop(machine.token, None)
            attempts = [a for a in self._attempts.values() if a.machine is machine]
        for attempt in attempts:
            self._lose(attempt)
        if why != "reconnected" and not self._stop.is_set():
            back = f"; its {len(attempts)} task(s) go back to the queue" if attempts else ""
            self._print.line(f"- {machine.name} {why}{back}")

    # -------------------------------------------------------------- watching

    def _count_sent(self, n: int) -> None:
        with self._lock:
            self._sent.append((time.monotonic(), n))

    def _send_rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            while self._sent and now - self._sent[0][0] > 10.0:
                self._sent.popleft()
            total = sum(n for _, n in self._sent)
        return total / 10.0

    def _monitor(self) -> None:
        last_status = 0.0
        while not self._stop.wait(1.0):
            now = time.monotonic()
            for machine in list(self._machines):
                if not machine.local and not machine.gone and now - machine.last_seen > HEARTBEAT_TIMEOUT:
                    self._leave(machine, "stopped responding")
            state = self._state
            if state is None or state.finished.is_set():
                continue
            self._check_warnings(state, now)
            if now - last_status >= self._print.interval:
                self._print.status(self._status_line(state))
                last_status = now

    def _status_line(self, state: _JobState) -> str:
        with self._lock:
            done = sum(t.state == "done" for t in state.tasks)
            failed = sum(t.state == "failed" for t in state.tasks)
            total = len(state.tasks)
            active = [m for m in self._machines if not m.gone]
            remaining = sum(t.task.weight for t in state.tasks if t.state in ("pending", "running"))
            throughput = 0.0
            for m in active:
                per = (0.0 if m.local or not m.fetch_rate else 1.0 / m.fetch_rate) + \
                      (1.0 / m.compute_rate if m.compute_rate else 0.0)
                if m.compute_rate:
                    throughput += 1.0 / per
        text = f"{state.job.label}: {done}/{total}"
        if failed:
            text += f" ({failed} failed)"
        text += f" | {len(active)} computer{'s' if len(active) != 1 else ''}"
        rate = self._send_rate()
        if rate > 0:
            text += f" | sending {format_bytes(rate)}/s"
        if throughput > 0 and remaining:
            text += f" | ~{format_seconds(remaining / throughput)} left"
        return text

    def _warn_once(self, key: str, text: str) -> None:
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        self._print.line(text)

    def _check_warnings(self, state: _JobState, now: float) -> None:
        if not self.helpers:
            return
        remote = [m for m in self._machines if not m.local and not m.gone]
        if not remote and now - state.started > NO_HELPERS_HINT:
            only = (" Nothing runs until one joins (local=False)." if self.local_machine is None else "")
            self._warn_once("no-helpers",
                            f"No helper has joined yet.{only} On each helper run: landschaft worker\n"
                            f"  If they do not appear: run `landschaft helpers firewall` here (Windows), or start "
                            f"them with: landschaft worker --connect {self._address()}:{self.port}")
        done = [m for m in remote if m.via.get("network")]
        n = sum(m.via.get("network", 0) for m in done)
        busy = sum(m.compute_s + m.starved_s for m in done)
        starved = sum(m.starved_s for m in done)
        if n >= 3 and busy > 0 and starved / busy > 0.25:
            self._warn_once("network",
                            f"! The network is the bottleneck: the helpers waited for their inputs "
                            f"{100 * starved / busy:.0f}% of the time (sending at ~{format_bytes(self._send_rate())}/s)."
                            " More helpers will not make this faster; a shared folder (NAS) or wired instead of "
                            "Wi-Fi connections will.")

    def _result(self, state: _JobState) -> JobResult:
        with self._lock:
            done = [t.task.name for t in state.tasks if t.state == "done"]
            failed = {t.task.name: (t.errors[-1] if t.errors else "failed") for t in state.tasks
                      if t.state == "failed"}
            machines = {m.name: m.done - state.counted.get(m.name, 0) for m in self._machines}
            machines = {k: v for k, v in machines.items() if v}
        return JobResult(done, failed, machines, time.monotonic() - state.started)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:80] or "task"


def _ema(old: float | None, new: float) -> float:
    return new if old is None or old == float("inf") else EMA * new + (1 - EMA) * old


def summary(label: str, result: JobResult) -> str:
    lines = [f"{label}: {len(result.done)} task(s) in {format_seconds(result.seconds)}"
             + (f", {len(result.failed)} failed" if result.failed else "")]
    if len(result.machines) > 1:
        lines.append("  " + ", ".join(f"{name}: {n}" for name, n in result.machines.items()))
    for name, error in result.failed.items():
        lines.append(f"  failed: {name}: {error.strip().splitlines()[-1] if error.strip() else error}")
    return "\n".join(lines)
