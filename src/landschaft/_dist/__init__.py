# SPDX-License-Identifier: GPL-3.0-or-later
"""Work shared with other computers of the local network (see :mod:`landschaft.parallel`).

Adapted from zeit's tile runner (``zeit/_tiles``): the same pairing, TLS, discovery and
queue, with registered operations of landschaft as the unit of work instead of a
command line run on a tile.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from . import _group
from ._coordinator import Coordinator, JobResult, summary
from ._tasks import InputFile, Job, Task

__all__ = ["Helpers", "InputFile", "Job", "JobResult", "Task", "helpers", "invite_helpers", "leave_group",
           "list_helpers", "open_firewall", "run_job", "run_worker"]


class Helpers:
    """This computer and the helpers of its group, for the ``on=`` argument of
    landschaft's functions that run many independent tasks.

    Give it to :func:`landschaft.batch`, :func:`landschaft.mspa_batch`,
    :func:`landschaft.simulate.scenarios` or :func:`landschaft.simulate.calibrate`
    (grid and random searches): their tasks run here and on the computers of this
    computer's group running ``landschaft worker``, with the same results as on this
    computer alone. Used in a ``with`` block, the helpers stay connected from one call
    to the next; otherwise each call starts and stops the coordinator.

    Parameters
    ----------
    local
        Run tasks on this computer too. ``local=False``: this computer only hands out
        tasks and collects their outputs.
    local_threads
        Cores of this computer for the tasks (default: all but one).
    port
        TCP port helpers connect to (the discovery broadcasts use UDP 47810); 0 picks a
        free one.
    discovery
        Answer the broadcasts of helpers looking for work on the local network (default:
        yes, except inside a batch job of Slurm, PBS or LSF, between whose nodes broadcasts
        do not pass). Without it, helpers connect with ``landschaft worker --connect <this
        computer>``.
    verbose
        Print who joined, the progress and failures.

    Examples
    --------
    >>> import landschaft as ls
    >>> with ls.helpers() as pool:                                         # doctest: +SKIP
    ...     runs = ls.batch("fad", "tiles/", "fad_out/", on=pool)
    ...     ens = ls.simulate.scenarios(model, start, drivers, replicates=100, seed=1, on=pool)
    """

    def __init__(self, *, local: bool = True, local_threads: int | None = None, port: int = _group.TCP_PORT,
                 discovery: bool | None = None, verbose: bool = True, out: Callable[[str], None] | None = None,
                 host: str = ""):
        if discovery is None:
            from .._jobs import in_job

            discovery = in_job() is None
        self._options = dict(local=local, local_threads=local_threads, port=port, discovery=discovery,
                             verbose=verbose, out=out, host=host)
        self._coordinator: Coordinator | None = None
        self._entered = False

    def _start(self) -> Coordinator:
        if self._coordinator is None:
            self._coordinator = Coordinator(helpers=True, **self._options).start()
        return self._coordinator

    @property
    def port(self) -> int:
        """The TCP port helpers connect to (known once started)."""
        return self._start().port

    @property
    def machines(self) -> list:
        """The computers working now: ``[{"name", "cores", "local", "tasks"}]``."""
        if self._coordinator is None:
            return []
        return [{"name": m.name, "cores": m.cores, "local": m.local, "tasks": m.done}
                for m in self._coordinator.machines]

    def run(self, job: Job) -> JobResult:
        """Run a :class:`Job` (one at a time) and wait for it."""
        coordinator = self._start()
        try:
            result = coordinator.run(job)
        finally:
            if not self._entered:
                self.close()
        if self._options["verbose"]:
            text = summary(job.label, result)
            (self._options["out"] or (lambda t: print(t, flush=True)))(text)
        return result

    def close(self) -> None:
        """Stop the coordinator: helpers go back to waiting for work."""
        if self._coordinator is not None:
            self._coordinator.close()
            self._coordinator = None

    def __enter__(self) -> "Helpers":
        self._entered = True
        self._start()
        return self

    def __exit__(self, *exc) -> None:
        self._entered = False
        self.close()

    def __repr__(self) -> str:
        state = "running" if self._coordinator is not None else "not started"
        return f"Helpers({state}, local={self._options['local']})"


def helpers(*, local: bool = True, local_threads: int | None = None, port: int = _group.TCP_PORT,
            discovery: bool | None = None, verbose: bool = True) -> Helpers:
    """This computer and the helpers of its group, for ``on=`` (see :class:`Helpers`).

    Parameters
    ----------
    local
        Run tasks on this computer too (``False``: only coordinate).
    local_threads
        Cores of this computer for the tasks (default: all but one).
    port
        TCP port helpers connect to.
    discovery
        Answer helpers that look for work on the local network (default: yes, except inside
        a batch job, where helpers connect with ``landschaft worker --connect``).
    verbose
        Print who joined, the progress and failures.

    Returns
    -------
    Helpers

    Examples
    --------
    >>> import landschaft as ls
    >>> with ls.helpers() as pool:                                     # doctest: +SKIP
    ...     runs = ls.batch("accounting", "tiles/", "out/", on=pool)
    """
    return Helpers(local=local, local_threads=local_threads, port=port, discovery=discovery, verbose=verbose)


def run_job(on, job: Job) -> JobResult:
    """Run ``job`` on ``on`` (a :class:`Helpers`)."""
    if not isinstance(on, Helpers):
        raise TypeError(f"on= takes ls.helpers(), not {type(on).__name__}")
    return on.run(job)


def run_worker(*, cores: int | None = None, connect: str | None = None, join: str | None = None,
               maps: Iterable[str] = (), cache_dir: str | None = None, cache_gb: float = 20.0) -> None:
    """Lend this computer to the group: wait for tasks and run them, until Ctrl+C.

    Parameters
    ----------
    cores
        Cores to lend (default: all but one).
    connect
        The coordinator's address (``"192.168.1.10"`` or ``"host:port"``), when the
        broadcasts of the local network do not reach it.
    join
        The invitation code shown by ``landschaft helpers invite``, the first time only.
    maps
        Where shared folders are mounted here, as ``"SRC=DST"``: a file the coordinator
        sees under SRC is read here under DST (``"Z:\\\\=/mnt/nas"``).
    cache_dir, cache_gb
        Where received input files are kept, and how much space they may take: the same
        data is not sent again.
    """
    from ._worker import run_worker as _run

    _run(cores=cores, connect=connect, join=join, maps=list(maps), cache_dir=cache_dir, cache_gb=cache_gb)


def invite_helpers(minutes: float = 10.0, *, port: int = _group.TCP_PORT) -> list:
    """Show an invitation code and let other computers join this one's group with
    ``landschaft worker --join <code>``, for ``minutes``. Returns the names of those that
    joined."""
    from ._pairing import Invitation

    invitation = Invitation(minutes=minutes, port=port)
    print(f"Invitation code: {invitation.code}  (valid for {minutes:g} minutes)\n"
          f"On each computer that will help, run:  landschaft worker --join {invitation.code}", flush=True)
    try:
        return invitation.wait()
    except KeyboardInterrupt:
        return list(invitation.joined)


def list_helpers(seconds: float = 5.0) -> list:
    """The helpers of this computer's group waiting for work now (busy ones are not
    looking): ``[{"name", "address", "cores", "version"}]``."""
    from ._pairing import list_helpers as _list

    return _list(seconds)


def leave_group() -> bool:
    """Forget this computer's group (it no longer helps nor uses helpers)."""
    return _group.forget_secret()


def open_firewall() -> str:
    """Let helpers reach this computer: the Windows firewall rules for the local network."""
    from ._system import open_firewall as _open

    return _open()
