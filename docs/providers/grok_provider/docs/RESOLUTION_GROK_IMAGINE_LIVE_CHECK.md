# Grok Imagine ACP live check

Date: 2026-06-20

Environment: `grok 0.2.54 (fee15ff8e)`, `grok agent stdio`, SuperGrok cached-token authentication.

The live `session/new` + `session/prompt` probe requested one generated red circle. ACP exposed `GenerateImage`, not `image_gen`:

- `tool_call.title`: `GenerateImage`
- `rawInput`: `description`, `filename`
- terminal status: `completed`
- `rawOutput`: `type=ImageGen`, `path`, `filename`, `session_folder`
- artifact: JPEG file at the reported local path
- observed generation time: approximately 4.2 seconds

The ACP tool inventory contained `GenerateImage` and did not contain `image_edit`, `EditImage`, `image_to_video`, or `reference_to_video` on grok 0.2.54. Candidate names are pre-registered in `IMAGINE_GROK_BUILTINS` for passthrough routing (not published to proxy MCP). Any name Grok CLI emits that is still unknown routes via `GrokBuiltin` without blocking the call.

Verification after implementation:

```text
GROK_INTEGRATION_TESTS=1 pytest -q grok_provider/tests/test_integration.py::test_generate_image_passthrough_round_trip
1 passed
```
