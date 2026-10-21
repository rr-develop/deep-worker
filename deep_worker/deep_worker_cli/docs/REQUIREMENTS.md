# Runtime requirements

## Python

`>=3.11,<4.0` (see `pyproject.toml`). Tested on 3.11–3.14.

## Operating system

Linux. The daemon model depends on Linux-specific facilities that have no
portable equivalent: `/proc/<pid>/stat` and `/proc/<pid>/cmdline` for process
identity, `inotify` for readiness events, and `setsid` for detaching
background servers. macOS and Windows are not supported; WSL2 is.

## Minimum kernel: Linux 4.19

**Linux 4.19 is the supported floor.** WSL2 is the reason it is this low:
Microsoft shipped 4.19-based kernels for a long time, and users who have not
updated their WSL image are still on `4.19.128-microsoft-standard`.

Everything on a user-facing path must therefore work on 4.19. Concretely,
these syscalls postdate the floor and **must not be required**:

| Syscall | Since | Notes |
| --- | --- | --- |
| `pidfd_send_signal` | 5.1 | |
| `clone3` | 5.3 | |
| `pidfd_open` | 5.3 | Also needs `<sys/pidfd.h>` at CPython build time — uv-installed interpreters are built without it, so this one can be missing on *any* kernel. |
| `waitid(P_PIDFD)` | 5.4 | |
| `openat2` | 5.6 | |
| `pidfd_getfd` | 5.6 | |
| `faccessat2` | 5.8 | |
| `close_range` | 5.9 | |
| `memfd_secret` | 5.14 | |

### Using them anyway

They may be used as an optimisation, never as a requirement: probe, catch
`OSError`/`AttributeError`, and fall back. A missing syscall answers
`ENOSYS` ("Function not implemented"), and an unguarded call turns that into
a failed command on a system that is otherwise fine.

A pidfd is the usual temptation, because it makes child-death detection
event-driven. The guarded pattern is in
`deep_worker_cli/process/web_daemon.py` (`_open_pidfd`, `_signal`): the web
server's readiness channel reports both outcomes on its own — bytes mean
ready, EOF means the child died — so the pidfd only ever sharpens cleanup.
`deep_worker_cli/lsp/service/supervisor.py` and
`deep_worker_cli/semantic_memory/service/supervisor.py` do the same.

Note that "missing" has two shapes, and a handler must accept both: the
kernel refusing the syscall is `OSError`/`ENOSYS`, while a CPython built
without `<sys/pidfd.h>` has no `os.pidfd_open` attribute at all, which is
`AttributeError`. Catching only the first still crashes on uv-installed
interpreters.

The fallback used when no pidfd is available — `waitpid` on a worker
thread, which stays event-driven from asyncio's point of view and does not
poll — lives in `deep_worker_cli/lsp/service/child_death.py`, shared by the
LSP supervisors. It deliberately imports nothing but `os`: it is reachable
from the LSP bridge, which `tests/unit_tests/lsp/test_bridge_thin_entry.py`
asserts loads none of the agent/ML stack.

For the regression that motivated this document, see
`tests/unit_tests/test_web_daemon.py` — `_ENOSYS_STUB` emulates the absence
of every post-4.19 call the web path touches, and
`test_web_path_uses_no_syscall_newer_than_the_supported_floor` fails if a new
one is introduced unguarded.

`tests/unit_tests/test_kernel_floor_guard.py` extends that from the web path
to every module in the package: it bans the post-floor calls that have no
fallback here, allows `pidfd_*` only in modules that handle its absence, and
checks each such call site individually — per site, because one correct
handler must not vouch for a narrowed one beside it.

## Why not simply require a newer kernel

The failure mode is bad out of proportion to the benefit: the user sees
`[Errno 38] Function not implemented` from a command that gives no hint which
call was missing, while the service it was starting is running fine. The
fallbacks are a few lines each.
