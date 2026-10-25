# SPDX-License-Identifier: GPL-3.0-or-later
"""A helper: lends this computer to the group. It waits for a coordinator, then pulls
tasks, runs them in child processes (only landschaft's registered operations) and sends
the outputs back.

While tasks run, the next one is already being fetched. An input file comes from the
helper's cache, from shared storage (a NAS both computers see), or over the network from
the coordinator, in that order of preference.
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from . import _group, _net, _ops, _system, _tasks
from ._slots import Slots
from ._tasks import format_bytes, format_seconds

HEARTBEAT = 5.0


def default_cache_dir() -> Path:
    """``helper`` in landschaft's cache (``LANDSCHAFT_CACHE``, else the platform's, see ``_download``), or
    the folder an older landschaft kept, where it exists."""
    from .._download import cache_dir

    if sys.platform == "win32":
        old = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "landschaft" / "helper-cache"
    else:
        old = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "landschaft" / "helper"
    new = cache_dir() / "helper"
    return old if not os.environ.get("LANDSCHAFT_CACHE") and old != new and old.is_dir() else new


def default_cores() -> int:
    from .._threads import default_threads

    return default_threads()


def info(cores: int) -> dict:
    import platform

    return {"name": _group.machine_name(), "version": _group.version(), "code": _group.code_fingerprint(),
            "cores": cores, "python": platform.python_version(), "platform": sys.platform}


def parse_maps(maps: Sequence[str]) -> list:
    """``SRC=DST`` pairs: a path the coordinator sees under SRC is read here under DST."""
    pairs = []
    for item in maps or ():
        if "=" not in item:
            raise ValueError(f"--map takes SRC=DST, got {item!r}")
        src, dst = item.split("=", 1)
        pairs.append((src, dst))
    return pairs


def map_path(path: str, maps: Sequence) -> list:
    """Where a coordinator's path may be found here: through each map, then as it is."""
    out = []
    norm = path.replace("\\", "/")
    for src, dst in maps:
        s = src.replace("\\", "/").rstrip("/")
        if norm.lower().startswith(s.lower()) and (len(norm) == len(s) or norm[len(s)] == "/"):
            rest = norm[len(s):].lstrip("/")
            out.append(os.path.join(dst, *rest.split("/")) if rest else dst)
    out.append(path)
    return out


def evict_cache(cache: Path, limit: int, keep: set) -> None:
    """Remove the least recently used files until the cache holds at most ``limit`` bytes
    (never those in ``keep``, nor files being written)."""
    try:
        files = [p for p in cache.iterdir() if p.is_file() and ".part" not in p.name]
    except OSError:
        return
    total = sum(p.stat().st_size for p in files)
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= limit:
            break
        if p in keep:
            continue
        try:
            size = p.stat().st_size
            p.unlink()
            total -= size
        except OSError:
            pass     # in use


def _target(connect: str) -> tuple:
    host, _, port = connect.rpartition(":") if connect.count(":") == 1 else (connect, "", "")
    return (host or connect), int(port) if port else _group.TCP_PORT


def run_worker(*, cores: int | None = None, connect: str | None = None, join: str | None = None,
               maps: Sequence[str] = (), cache_dir: str | None = None, cache_gb: float = 20.0,
               once: bool = False, stop: threading.Event | None = None,
               out: Callable[[str], None] | None = None) -> None:
    """Lend this computer to the group until ``stop`` is set (or Ctrl+C). ``join``: the
    invitation code from ``landschaft helpers invite``, for the first time. ``connect``: the
    coordinator's address (host or host:port), when broadcasts do not reach it. ``once``:
    return after the first coordinator is gone."""
    from . import _pairing

    out = out or (lambda text: print(text, flush=True))
    stop = stop or threading.Event()
    cores = cores if cores and cores > 0 else default_cores()
    pairs = parse_maps(maps)
    if join:
        secret = _pairing.join(join, connect=connect, out=out, stop=stop)
        _group.save_secret(secret)
        out("Joined the group. From now on, just run: landschaft worker")
    secret = _group.load_secret()
    if secret is None:
        raise RuntimeError("this computer is not in a group yet. On the main computer run `landschaft helpers "
                           "invite`, then here: landschaft worker --join <code>")
    gid = _group.group_id(secret)
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    me = info(cores)
    backoff: dict = {}
    unreachable = False
    out(f"{me['name']} lends {cores} cores to the group (landschaft {me['version']}). Waiting for work; "
        "Ctrl+C to stop.")
    try:
        while not stop.is_set():
            if connect:
                host, port = _target(connect)
            else:
                now = time.monotonic()
                found = _net.find({"kind": "helper", "group": gid, "name": me["name"], "cores": cores,
                                   "version": me["version"]},
                                  lambda r: r.get("kind") == "coordinator" and r.get("group") == gid,
                                  stop=stop)
                if found is None:
                    continue
                host, reply = found
                port = int(reply.get("port") or _group.TCP_PORT)
                if backoff.get(host, 0) > now:
                    stop.wait(2.0)
                    continue
            session = _net.Session(host, port)
            try:
                welcome = session.handshake(lambda salt: secret, me)
            except _net.ProtocolError as e:
                out(f"{host}: {e}")
                backoff[host] = time.monotonic() + 60
                session.close()
                if once:
                    raise
                stop.wait(30 if connect else 2)
                continue
            except OSError as e:
                session.close()
                if connect:
                    if not unreachable:
                        out(f"Cannot reach {host}:{port} ({e}); trying every 10 s")
                    unreachable = True
                    stop.wait(10)
                continue
            unreachable = False
            out(f"Working for {welcome.get('name', host)} ({host})")
            job = _Job(session, cores, cache, int(cache_gb * 1e9), pairs, stop, out)
            with _system.keep_awake():
                job.run()
            if once:
                return
            out("Waiting for work.")
    except KeyboardInterrupt:
        stop.set()
        out("Stopped.")


class _Job:
    """Work for one coordinator, until it closes or goes away."""

    def __init__(self, session: _net.Session, cores: int, cache: Path, cache_bytes: int, maps: Sequence,
                 stop: threading.Event, out: Callable[[str], None]):
        self.session, self.cores, self.cache, self.cache_bytes = session, cores, cache, cache_bytes
        self.maps, self.stop, self.out = maps, stop, out
        self.ready: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.held: dict = {}           # attempt -> its cancel event
        self.waiting = 0               # tasks fetched (or being fetched) and not started
        self.used = 0                  # threads of the running tasks
        self.ended = threading.Event()
        self.error: str | None = None
        self.tmp = Path(tempfile.mkdtemp(prefix="landschaft-worker-"))
        self.slots = Slots(cores, self.tmp / "children")
        self.results_roots: dict = {}
        self.uses: dict = {}           # cached file -> tasks holding it (never evicted meanwhile)

    def run(self) -> None:
        threads = [threading.Thread(target=self._fetcher, name="landschaft-worker-fetch", daemon=True),
                   threading.Thread(target=self._heartbeat, name="landschaft-worker-heartbeat", daemon=True)]
        for t in threads:
            t.start()
        try:
            self._runner()
        except KeyboardInterrupt:
            self.stop.set()
            raise
        finally:
            self.ended.set()
            with self.lock:
                for event in self.held.values():
                    event.set()
                self.cond.notify_all()
            if self.stop.is_set():
                try:
                    self.session.call("POST", "/bye", {})
                except Exception:  # noqa: BLE001, S110
                    pass
            for t in threads:
                t.join(timeout=10)
            with self.lock:
                while self.used:       # running tasks see the cancel and stop
                    self.cond.wait(1.0)
            self.slots.close()
            self.session.close()
            shutil.rmtree(self.tmp, ignore_errors=True)
        if self.error:
            self.out(f"Lost the coordinator: {self.error}")

    def _end(self, error: str | None = None) -> None:
        if error and not self.ended.is_set():
            self.error = error
            with self.lock:
                for event in self.held.values():    # the coordinator is gone: stop working for it
                    event.set()
        self.ended.set()
        self.ready.put(None)
        with self.lock:
            self.cond.notify_all()

    # -------------------------------------------------------------- fetching

    def _fetcher(self) -> None:
        s = self.session.clone()
        try:
            while not self.ended.is_set() and not self.stop.is_set():
                with self.lock:
                    if self.waiting:               # one task ahead at most
                        self.cond.wait(0.5)
                        continue
                    self.waiting += 1
                try:
                    status, reply = s.call("POST", "/next", {})
                except Exception as e:  # noqa: BLE001
                    self._unwait()
                    self._end(f"{type(e).__name__}: {e}")
                    return
                if status != 200:
                    self._unwait()
                    self._end(reply.get("error") or f"HTTP {status}")
                    return
                if reply.get("done"):
                    self._unwait()
                    self._end()
                    return
                if "task" not in reply:
                    self._unwait()
                    self.stop.wait(min(float(reply.get("wait") or 3), 10.0))
                    continue
                task = reply["task"]
                aid = str(task.get("attempt"))
                with self.lock:
                    self.held[aid] = threading.Event()
                try:
                    if task.get("op") not in _ops.OPS:
                        raise ValueError(f"unknown operation {task.get('op')!r}: is landschaft the same version?")
                    start = time.monotonic()
                    folder, via, files = self._prepare(s, task)
                    self.ready.put((task, folder, via, start, time.monotonic(), files))
                except Exception as e:  # noqa: BLE001
                    self._unwait()
                    if not self.ended.is_set():
                        transfer = isinstance(e, (OSError, _net.ProtocolError, http.client.HTTPException))
                        self._report_failure(s, aid, f"could not get the inputs here: {type(e).__name__}: {e}",
                                             transfer=transfer)
                    self._release(aid)
        finally:
            s.close()
            if not self.ended.is_set():
                self._end()

    def _unwait(self) -> None:
        with self.lock:
            self.waiting -= 1
            self.cond.notify_all()

    def _prepare(self, s: _net.Session, task: dict) -> tuple:
        """The task's input folder, each file at its name, where the files came from and the
        cached files it holds."""
        aid = str(task["attempt"])
        folder = self.tmp / f"task-{_safe(aid)}" / "in"
        folder.mkdir(parents=True, exist_ok=True)
        vias = set()
        held: list = []
        moved, seconds = 0, 0.0
        for spec in task.get("files", []):
            name = _tasks.check_name(spec["name"])
            key = str(spec["key"])
            if not key.isalnum():
                raise ValueError("bad file key")
            suffix = "".join(c for c in Path(name).suffix if c.isalnum() or c == ".")[:12]
            cached = self.cache / f"{key}{suffix}"     # complete: files only get there by a rename
            start = time.monotonic()
            if cached.is_file():
                os.utime(cached)
                via = "cache"
            else:
                part = self.cache / f"{key}.{_safe(aid)}.part{suffix}"
                try:
                    source = self._shared_source(spec)
                    if source is not None:
                        shutil.copyfile(source, part)
                        via = "shared"
                    else:
                        s.download(f"/file/{aid}/{int(spec['index'])}", part)    # checks the length it was told
                        via = "network"
                    os.replace(part, cached)
                finally:
                    if part.exists():
                        part.unlink()
                moved += cached.stat().st_size
                seconds += time.monotonic() - start
            vias.add(via)
            dest = folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(cached, dest)
            except OSError:
                shutil.copyfile(cached, dest)
            with self.lock:
                self.uses[cached] = self.uses.get(cached, 0) + 1
            held.append(cached)
        if moved:
            where = "from shared storage" if vias == {"shared"} else "from the network" if "network" in vias else ""
            self.out(f"> {task.get('name')}: {format_bytes(moved)} {where} in {format_seconds(seconds)}")
        with self.lock:
            keep = set(self.uses)
        evict_cache(self.cache, self.cache_bytes, keep)
        via = "network" if "network" in vias else "shared" if "shared" in vias else "cache" if vias else "none"
        return folder, via, held

    def _drop_uses(self, files) -> None:
        for f in files:
            n = self.uses.get(f, 0) - 1
            if n > 0:
                self.uses[f] = n
            else:
                self.uses.pop(f, None)

    def _shared_source(self, spec: dict) -> str | None:
        path = str(spec.get("path") or "")
        if not path:
            return None
        for candidate in map_path(path, self.maps):
            try:
                if (os.path.isfile(candidate) and os.path.getsize(candidate) == int(spec["size"])
                        and _tasks.probe(candidate) == spec["probe"]):
                    return candidate
            except OSError:
                continue
        return None

    # -------------------------------------------------------------- running

    def _runner(self) -> None:
        first = True
        waiting_since = time.monotonic()
        while True:
            try:
                item = self.ready.get(timeout=0.5)    # a timeout keeps Ctrl+C working on Windows
            except queue.Empty:
                continue
            if item is None:
                return
            task, folder, via, fetch_start, fetch_end, files = item
            aid = str(task["attempt"])
            threads = max(1, min(int(task.get("threads") or self.cores), self.cores))
            with self.lock:
                starved = 0.0 if first or self.used else max(0.0, fetch_end - max(waiting_since, fetch_start))
                first = False
                while self.used and self.used + threads > self.cores and not self.ended.is_set():
                    self.cond.wait(0.5)
                self.waiting -= 1
                cancel = self.held.get(aid)
                if cancel is None or cancel.is_set() or (self.ended.is_set() and self.error):
                    self.cond.notify_all()
                    self._release_locked(aid)
                    self._drop_uses(files)
                    shutil.rmtree(folder.parent, ignore_errors=True)
                    waiting_since = time.monotonic()
                    continue
                self.used += threads
                self.cond.notify_all()
            threading.Thread(target=self._run_one, args=(task, folder, via, starved, cancel, fetch_end - fetch_start,
                                                         threads, files), name="landschaft-worker-task",
                             daemon=True).start()
            waiting_since = time.monotonic()

    def _run_one(self, task: dict, inputs: Path, via: str, starved: float, cancel: threading.Event, fetch_s: float,
                 threads: int, files: list) -> None:
        aid, name = str(task["attempt"]), str(task.get("name"))
        out_dir = inputs.parent / "out"
        s = self.session.clone()
        slot = self.slots.acquire()
        try:
            self.out(f"  {name}: {task.get('label')} on {threads} core{'s' if threads > 1 else ''}")
            ok, seconds, message, stats = slot.run(task["op"], task["params"], inputs, out_dir, threads, cancel)
            if ok:
                stats = {"compute_s": round(seconds, 3), "fetch_s": round(fetch_s, 3),
                         "starved_s": round(starved, 3), "via": via, **{k: v for k, v in stats.items()
                                                                        if k not in ("compute_s", "fetch_s")}}
                placed = self._place(task, out_dir)
                if placed is not None:     # in the results folder already: only say so
                    status, reply = s.call("PUT", f"/result/{aid}", body=b"", headers={
                        "X-Landschaft-Placed": placed, "X-Landschaft-Stats": json.dumps(stats)})
                else:
                    archive = inputs.parent / "out.tar"
                    size = _tasks.pack(out_dir, archive)
                    with open(archive, "rb") as f:
                        status, reply = s.call("PUT", f"/result/{aid}", body=f, headers={
                            "Content-Length": str(size), "Content-Type": "application/x-tar",
                            "X-Landschaft-Stats": json.dumps(stats)})
                if status == 200 and reply.get("accepted"):
                    self.out(f"< {name}: done in {format_seconds(seconds)}")
                elif status == 200:
                    self.out(f"< {name}: done in {format_seconds(seconds)}, but another computer was faster")
                else:
                    self.out(f"! {name}: the outputs were refused: {reply.get('error') or status}")
            elif cancel.is_set():
                if not self.ended.is_set():
                    self.out(f"  {name}: cancelled (another computer finished it)")
            else:
                lines = message.strip().splitlines()
                self.out(f"! {name}: failed: {lines[-1] if lines else 'failed'}")
                self._report_failure(s, aid, message)
        except (OSError, _net.ProtocolError, http.client.HTTPException) as e:
            self._end(f"{type(e).__name__}: {e}")
        finally:
            self.slots.release(slot)
            s.close()
            shutil.rmtree(inputs.parent, ignore_errors=True)
            with self.lock:
                self.used -= threads
                self._release_locked(aid)
                self._drop_uses(files)
                self.cond.notify_all()

    def _place(self, task: dict, out_dir: Path) -> str | None:
        """Copy the outputs into the coordinator's results folder when this computer sees it
        (a NAS): the name of the folder written, or None to send them instead."""
        spec = task.get("results")
        if not isinstance(spec, dict):
            return None
        name = str(spec.get("name") or "")
        if not name or Path(name).name != name or name in (".", ".."):
            return None
        root = self._results_root(spec)
        if root is None:
            return None
        dest = Path(root) / name
        try:
            shutil.copytree(out_dir, dest)
        except OSError as e:
            shutil.rmtree(dest, ignore_errors=True)
            self.out(f"  could not write into the shared results folder ({e}); sending the outputs")
            return None
        return name

    def _results_root(self, spec: dict) -> str | None:
        key = (str(spec.get("path")), str(spec.get("token")))
        if key not in self.results_roots:
            found = None
            probe = Path(str(spec.get("probe") or "")).name
            for candidate in map_path(key[0], self.maps) if probe else ():
                try:
                    if (Path(candidate) / probe).read_text().strip() == key[1]:
                        found = candidate
                        break
                except OSError:
                    continue
            self.results_roots[key] = found
        return self.results_roots[key]

    def _report_failure(self, s: _net.Session, aid: str, message: str, transfer: bool = False) -> None:
        try:
            s.call("POST", f"/fail/{aid}", {"error": message[-4000:], "transfer": transfer})
        except Exception:  # noqa: BLE001, S110
            pass

    def _release(self, aid: str) -> None:
        with self.lock:
            self._release_locked(aid)

    def _release_locked(self, aid: str) -> None:
        self.held.pop(aid, None)

    # -------------------------------------------------------------- heartbeat

    def _heartbeat(self) -> None:
        s = self.session.clone()
        misses = 0
        try:
            while not self.ended.wait(HEARTBEAT):
                with self.lock:
                    held = list(self.held)
                try:
                    status, reply = s.call("POST", "/heartbeat", {"held": held})
                except Exception as e:  # noqa: BLE001
                    misses += 1
                    if misses >= 3:
                        self._end(f"no answer ({type(e).__name__}: {e})")
                        return
                    continue
                if status == 401:
                    self._end("the coordinator no longer knows this computer")
                    return
                misses = 0
                with self.lock:
                    for aid in reply.get("cancel", []):
                        event = self.held.get(str(aid))
                        if event is not None:
                            event.set()
        finally:
            s.close()


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:80] or "task"
