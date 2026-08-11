# Building and Installing deep-worker from a Wheel

This guide explains how to build a PyPI-ready distribution of `deep-worker`
from a source checkout, verify it, and install it into a clean environment.

> **Scope:** building, checking, and installing. Publishing to PyPI is **not**
> covered here and is intentionally left to the release process.

## 1. Background: source checkout vs. installed wheel

Until now the project was used **only from a source checkout** — the README
explicitly says *"no `pip install`"* and launchers are symlinked into `PATH`.
That remains the recommended workflow for **developing** deep-worker itself.

Starting with this release the package can also be **built and installed as a
regular Python distribution** (`pip install deep-worker`). The two workflows
are complementary:

| Workflow | When to use |
|---|---|
| Source checkout (symlinked launchers) | Developing deep-worker, debugging, full feature set |
| Installed wheel (`pip install deep-worker`) | Consuming deep-worker as a dependency / on machines without the source tree |

Note that the *source checkout* workflow still does **not** require any
`pip install` — nothing about the existing setup changes for existing users.

## 2. Prerequisites

- Python 3.12+ (`python3` on the build machine)
- `build` (PEP 517 frontend): `python3 -m pip install build` in the build
  environment, or use `uv` if available
- `twine` for `twine check` (recommended; `uvx twine` also works)
- Network access: the build uses build isolation (hatchling) and `pip install`
  resolves dependencies from PyPI

## 3. Build

From the package root (`deep_worker/deep_worker_cli`):

```bash
cd deep_worker/deep_worker_cli
python3 -m build
```

This produces two artifacts in `dist/`:

- `deep_worker-<version>-py3-none-any.whl` — the wheel
- `deep_worker-<version>.tar.gz` — the sdist

The sdist is **self-contained**: it embeds the vendored `deepagents/` fork and
the `LICENSE` file, so a wheel can also be rebuilt from the extracted sdist
without the monorepo:

```bash
cd dist
tar xzf deep_worker-<version>.tar.gz
cd deep_worker-<version>
python3 -m build --wheel
```

## 4. Check

Verify both artifacts:

```bash
twine check dist/*
```

`twine check` must report **PASSED** for both the wheel and the sdist.

The generated `METADATA` must satisfy the following constraints (checked by
the integration tests in `tests/integration_tests/test_pypi_build.py`):

- **No** `file://` or any direct-reference (`@`) `Requires-Dist` lines;
- **No** `Requires-Dist: deepagents` — `deepagents` is a fork of LangChain's
  `deepagents` and is **vendored into this package**, not a dependency;
- `Requires-Dist: rpyc-async>=1.0.0` is present (the runtime RPyC
  implementation now comes from the public PyPI package, not a local path);
- `langgraph==1.0.8` and `langgraph-prebuilt==1.0.7` are pinned (see §7 —
  newer `langgraph-prebuilt` releases import a symbol that does not exist in
  `langgraph` 1.0.x, breaking `import deepagents`).

## 5. Install into a clean environment

Install the wheel into a fresh virtualenv **without** the source tree on
`PYTHONPATH`:

```bash
python3 -m venv .venv-check
source .venv-check/bin/activate
pip install dist/*.whl
```

Smoke-test the entry points (with the environment activated):

```bash
deep-worker --help      # exit 0
dwcli --help            # exit 0
deep-worker-cli --help  # exit 0
```

And the vendored fork + rpyc API used by the runtime:

```bash
python -c "from rpyc_async.core.protocol import DEFAULT_CONFIG; from rpyc_async.utils.async_server import AsyncioServer; print('rpyc OK')"
python -c "from deepagents.graph import create_deep_agent; print('deepagents OK')"
```

## 6. Known limitations of the installed wheel

The wheel is intentionally a **subset** of the monorepo. The following are
**not** available from a wheel install and are expected to remain so for now:

### 6.1 `--import-claude` is unavailable

`deep-worker --import-claude` requires the `claude_code_provider` package,
which lives **only in the source checkout** and is **not** shipped in the
wheel. On an installed wheel the flag exits with code 2 and a clear message:

```
--import-claude requires the claude_code_provider package, which is only
available in the source checkout (not in the PyPI wheel).
```

Every entry-point help path (`deep-worker --help`, `dwcli --help`,
`deep-worker-cli --help`) works without `claude_code_provider`. Commands
that do not reach the documented lazy `tools.py` dependency (see §6.2)
remain available; the full feature set is only guaranteed from the source
checkout.

### 6.2 Agent startup with `web_search` tools

`tools.py` also hard-requires `claude_code_provider` for certain
`web_search`-related tools. That import is **lazy** (it happens only when an
agent is actually created), so entry points and `--help` are unaffected, but
starting an agent from an installed wheel with those tools may fail with a
missing-module error. This is a documented limitation of the installed-wheel
workflow; the source checkout remains the full-featured path. A future
iteration may make these tools optional.

## 7. Important: do not co-install the official `deepagents`

This package **vendors its own fork of `deepagents`** as a top-level
`deepagents` package inside the wheel. The public PyPI project named
`deepagents` is **a different codebase**, and co-installing both in the same
environment is **not supported** (last-writer-wins on the shared top-level
name). There is deliberately **no** separate distribution for the fork — it is
shipped as part of `deep-worker`. If you need `deepagents`, install
`deep-worker` and use its vendored copy.

## 8. Local development (`uv`)

For development inside the monorepo, `deepagents` is provided via a
`[dependency-groups] test` group + `[tool.uv.sources]` path override (PEP 735),
so it never leaks into the built wheel's METADATA:

```bash
uv sync --all-groups          # installs the test group incl. local deepagents
uv run --all-groups ty check deep_worker_cli
uv run --group test pytest
```

## 9. Publishing

Publishing to PyPI is **out of scope** for this guide (see §1). When the time
comes, the release process will upload `dist/*` after `twine check` passes —
nothing in this guide performs a publish.

## 10. Verification

The integration suite in `tests/integration_tests/test_pypi_build.py`
(`-m integration`) automates §3–§5 for both the monorepo-built wheel and the
sdist-derived wheel. Run it with:

```bash
python3 -m pytest tests/integration_tests/test_pypi_build.py -m integration
```
