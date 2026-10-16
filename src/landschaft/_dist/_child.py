# SPDX-License-Identifier: GPL-3.0-or-later
"""A child process that runs tasks: ``python -m landschaft._dist._child``.

It reads one request per line on stdin (JSON: the operation, its parameters, the input
and output folders, the threads) and answers one line on the protocol channel (the
original stdout; whatever the operations print goes to stderr). A crash costs the task,
not the helper.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    import landschaft  # noqa: F401  (loaded once, before the first task)
    from landschaft._dist import _ops

    def say(message: dict) -> None:
        channel.write(json.dumps(message) + "\n")
        channel.flush()

    say({"ready": True, "pid": os.getpid()})
    for line in sys.stdin:
        if not line.strip():
            continue
        start = time.perf_counter()
        try:
            req = json.loads(line)
            stats = _ops.run(req["op"], req["params"], Path(req["inputs"]), Path(req["out"]), int(req["threads"]))
            say({"ok": True, "seconds": time.perf_counter() - start, "stats": stats})
        except BaseException as e:  # noqa: BLE001  (reported, the process carries on)
            if isinstance(e, KeyboardInterrupt):
                raise
            tail = "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:])
            say({"ok": False, "seconds": time.perf_counter() - start, "error": f"{type(e).__name__}: {e}",
                 "traceback": tail[-4000:]})


if __name__ == "__main__":
    main()
