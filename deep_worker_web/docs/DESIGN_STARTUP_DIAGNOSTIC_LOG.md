# Startup Diagnostic Log for deep_worker_web (a related issue)

## Goal

At every deep_worker_web server start (run.py, python -m app.main, uvicorn,
gunicorn) write a one-line [STARTUP] diagnostic to the server's rotating log.

## Format

One line, grep-able key=value pairs:

```
[STARTUP] component=deep_worker_web host=127.0.0.1 port=7890
commit=2176b8d describe=2176b8d-dirty tree=/path/to/tasks
dev_tree=false pid=1234567 debug=false reload=false
DEEP_WORKER_WEB_DEBUG=1 DEEP_WORKER_WEB_RELOAD=
```

## Where to inject

A single point covers all startup paths: the FastAPI lifespan context manager
in `app/main.py:lifespan()`. Both `run.py` and `python -m app.main` call
`uvicorn.run("app.main:app", ...)`, which triggers the lifespan.

## What fields

| Field | Source |
|-------|--------|
| component | Literal `"deep_worker_web"` |
| host | `settings.host` |
| port | `settings.port` |
| commit | `git -C <tree_root> rev-parse HEAD` (or `"unknown"`) |
| describe | `git -C <tree_root> describe --always --dirty` (or fallback to commit) |
| tree | `_CHECKOUT_ROOT` (= `settings.checkout_root`) |
| dev_tree | `"DEEP_WORKER_DEV_TREE"` env var truthiness |
| pid | `os.getpid()` |
| debug | `settings.debug` |
| reload | `settings.reload` |
| DEEP_WORKER_WEB_* | All `DEEP_WORKER_WEB_*` env vars that are set |

## Files to modify

1. `app/config.py` — add module-level `_get_git_commit_info()` helper
2. `app/main.py` — add `[STARTUP] logger.info(...)` at the top of `lifespan()`

## Error handling

If `git` is not available or the tree has no `.git`, both `commit` and
`describe` become `"unknown"`. The server must not crash.

## Out of scope

- Agent-side logging (a related issue already covers agents)
- Changing the session_metadata format
- Changing existing CLI behavior
