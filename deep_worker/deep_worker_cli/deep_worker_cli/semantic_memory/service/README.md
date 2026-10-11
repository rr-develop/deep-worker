# deep-worker shared semantic-memory service

See `docs/POLICY_NO_INPROCESS_HEAVY_MCP.md` and
`docs/DESIGN_SEMANTIC_MEMORY_SHARED_SERVICE.md` for the architecture,
sign-off, and commit sequence.

This service exists so the embedding stack (`llama-index` + LanceDB +
`HuggingFaceEmbedding` → `transformers` → `torch`) is loaded **once per OS
user** instead of once per agent. Measured on this host 2026-08-30:
7.13 s import, 884 MB RSS — against policy thresholds of 3 s / 200 MB.

Unit of sharing is the **host**, not the project (design §2): the dominant
resident is the embedding model, and it is identical for every repository
and every agent. Per-project sharding would reload it N times and recreate
the very problem being removed.

```
semantic_memory/
├── __init__.py
├── registry.py            — state.json reader/writer (no grpc, no engine)
└── service/
    ├── __init__.py        — MUST STAY EMPTY (see below)
    ├── README.md          — this file
    ├── supervisor.py      — auto-start, flock, detached spawn, readiness
    └── proto/
        ├── semantic_memory.proto        — gRPC contract (source of truth)
        ├── semantic_memory_pb2.py       — generated stubs (COMMITTED)
        ├── semantic_memory_pb2.pyi      — static type stubs (COMMITTED)
        └── semantic_memory_pb2_grpc.py  — generated stubs (COMMITTED)
```

## `service/__init__.py` must stay empty

Client lightness rests on it. The supervisor lives *inside* `service/`
(design §9.1) because policy §8 allows only `*.service*` packages to spawn
a backend — but the agent-side client imports the supervisor, so anything
executed by `service/__init__.py` would land in every agent process.
Keeping it empty means importing `service.supervisor` cannot drag in
`service.grpc_server` and the engine behind it.

This is the defect `repo_graph` has: its design doc §3.3 states the client
does not import `repo_graph.service`, while `stdio_server.py` does exactly
that — benign only by accident. A test pins the emptiness here.

## Pinned toolchain (design §4)

| Component | Version | Rationale |
|-----------|---------|-----------|
| `grpcio`         | `1.76.0` | Host-installed, `grpc.aio` (async server + client). |
| `grpcio-tools`   | `1.76.0` | Same MAJOR/MINOR as `grpcio` — mandatory. |
| `protobuf` (runtime) | `6.33.6` | Installed on the host; gencode/runtime must match by MAJOR. |
| `protobuf` (gencode) | `6.31.1` | **Bundled inside `grpcio-tools 1.76.0`** — this is what the generated header reports, and it is expected to differ from the runtime version. `lsp_service_pb2.py` reports the same `6.31.1`. |

**Do NOT install `grpcio-tools` from a MINOR series different from
`grpcio`.** `grpcio-tools >= 1.83` bundles `protobuf 7.x`, which requires
runtime protobuf ≥ 7 — that conflicts with several other host packages
(`grpcio-status`, `google-api-core`, `google-cloud-firestore`). Keeping
both grpc packages on `1.76.x` matches the installed `grpcio 1.76.0` and
preserves the rest of the environment.

## Regenerating stubs

```
# run from the repository root
python3 -m grpc_tools.protoc \
  -I deep_worker/deep_worker_cli/deep_worker_cli/semantic_memory/service/proto \
  --python_out=deep_worker/deep_worker_cli/deep_worker_cli/semantic_memory/service/proto \
  --pyi_out=deep_worker/deep_worker_cli/deep_worker_cli/semantic_memory/service/proto \
  --grpc_python_out=deep_worker/deep_worker_cli/deep_worker_cli/semantic_memory/service/proto \
  semantic_memory.proto

# grpc_tools generates absolute imports; rewrite the sibling import so the
# stubs work as a real package:
sed -i 's/^import semantic_memory_pb2 as /from . import semantic_memory_pb2 as /' \
  deep_worker/deep_worker_cli/deep_worker_cli/semantic_memory/service/proto/semantic_memory_pb2_grpc.py

# Re-apply the enum suppression, or the next commit is blocked (see below):
sed -i -E 's/^(class (SessionEventType|IndexKind)\(int, metaclass=_enum_type_wrapper\.EnumTypeWrapper\):)$/\1  # ty: ignore[conflicting-metaclass]  # generated .pyi: EnumTypeWrapper+int base is standard proto3 pattern/' \
  deep_worker/deep_worker_cli/deep_worker_cli/semantic_memory/service/proto/semantic_memory_pb2.pyi
```

**Both `sed` steps are mandatory after every regeneration.** `protoc`
emits each enum as `class E(int, metaclass=EnumTypeWrapper)`, which `ty`
rejects with `conflicting-metaclass`: `EnumTypeWrapper` is not a subclass
of `type`, the metaclass of the `int` base. The pattern is what protobuf
generates for every proto3 enum and cannot be fixed in the `.proto`, so
the suppression is the only option — the pre-commit type check is a hard
gate and `--no-verify` is forbidden. `lsp_service_pb2.pyi` carries the
same comment; it just never documented the step, which is how this
recurs.

The `--pyi_out` step is REQUIRED — the `.pyi` stub is what makes the
generated dynamic message classes visible to `ty` / `pyright`. Without it
the pre-commit type check reports `unresolved-attribute` for every
`HealthRequest` / `SessionEvent` / … lookup on `semantic_memory_pb2`.

The stubs are committed to git (policy §7).
