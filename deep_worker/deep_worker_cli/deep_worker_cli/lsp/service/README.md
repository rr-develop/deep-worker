# deep-worker LSP shared service (per-project, LSP-agnostic)

See `docs/POLICY_NO_INPROCESS_HEAVY_MCP.md` and
`docs/DESIGN_LSP_SHARED_SERVICE.md` for the architecture, sign-off, and
commit sequence.

Package layout (grows over subsequent commits — this file is the first
placeholder that pins the toolchain):

```
lsp/service/
├── __init__.py
├── README.md              — this file
└── proto/
    ├── __init__.py
    ├── lsp_service.proto        — gRPC contract (source of truth)
    ├── lsp_service_pb2.py       — generated stubs (COMMITTED)
    ├── lsp_service_pb2.pyi      — static type stubs (COMMITTED)
    └── lsp_service_pb2_grpc.py  — generated stubs (COMMITTED)
```

## Pinned toolchain (design §4)

| Component | Version | Rationale |
|-----------|---------|-----------|
| `grpcio`         | `1.76.0` | Host-installed, `grpc.aio` (async server + client). |
| `grpcio-tools`   | `1.76.0` | Same MAJOR/MINOR as `grpcio` — mandatory. |
| `protobuf`       | `6.33.6` | Bundled with `grpcio-tools 1.76`; gencode/runtime must match by MAJOR. |
| `libprotoc`      | `31.1`   | Shipped inside `grpcio-tools 1.76.0`; do not install a separate `protoc`. |
| `watchfiles`     | `1.1.1`  | inotify wrapper (used by change-detector in a later commit). |

**Do NOT install `grpcio-tools` from a MINOR series different from
`grpcio`.** `grpcio-tools >= 1.83` bundles `protobuf 7.x`, which requires
runtime protobuf ≥ 7 — that conflicts with several other host packages
(`grpcio-status`, `google-api-core`, `google-cloud-firestore`). Keeping
both grpc packages on `1.76.x` matches the currently installed
`grpcio 1.76.0` and preserves the rest of the environment.

## Regenerating stubs

```
cd /path/to/repo
python3 -m grpc_tools.protoc \
  -I deep_worker/deep_worker_cli/deep_worker_cli/lsp/service/proto \
  --python_out=deep_worker/deep_worker_cli/deep_worker_cli/lsp/service/proto \
  --pyi_out=deep_worker/deep_worker_cli/deep_worker_cli/lsp/service/proto \
  --grpc_python_out=deep_worker/deep_worker_cli/deep_worker_cli/lsp/service/proto \
  lsp_service.proto

# grpc_tools generates absolute imports; rewrite the sibling import so the
# stubs work as a real package:
sed -i 's/^import lsp_service_pb2 as /from . import lsp_service_pb2 as /' \
  deep_worker/deep_worker_cli/deep_worker_cli/lsp/service/proto/lsp_service_pb2_grpc.py
```

The `--pyi_out` step is REQUIRED — the `.pyi` stub is what makes the
generated dynamic message classes visible to `ty` / `pyright`. Without
it the pre-commit type check reports `unresolved-attribute` for every
`HealthRequest` / `SessionEvent` / … lookup on `lsp_service_pb2`.

The stubs are committed to git (policy §7 / design §16 Q2). A pre-commit
invariant that will land in a later commit blocks `.proto` changes without
matching stub regeneration.
