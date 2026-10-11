#!/usr/bin/env python3
"""Fake ACP CLI for integration tests.

Speaks a minimal JSON-RPC subset of the Agent Client Protocol over stdio so
the real ``AcpClient`` (and the whole channel) can be exercised end-to-end
against a controllable subprocess — no real Claude Code needed.

Scenarios (env ``FAKE_ACP_SCENARIO``):

  text        — session/prompt streams agent text then ends the turn.
  tool        — session/prompt emits a built-in tool_call, a completed
                tool_call_update, then ends the turn.
  tool-proxy  — channel-A signed gate: session/prompt emits a proxy-MCP
                tool_call (mcp__<server>__<tool>) and BLOCKS on a real HTTP
                tools/call to the in-agent proxy MCP server until the handler
                is resolved by proxy_mcp.resolve_call; then ends the turn.
                The tools/call reply is written to FAKE_ACP_RESULT_FILE and
                every session/prompt is appended to FAKE_ACP_COUNT_FILE so
                tests can assert the single-prompt invariant.
  tool-then-end
              — emits a proxy-MCP tool_call AND ends the turn in the same
                batch (no tools/call block), so the update and the prompt
                reply land on the same event loop iteration. Regression
                harness for the _finalize_response tool_call drop.
  turn-ending-open
              — turn 1: emits a proxy-MCP tool_call and LEAVES session/prompt
                UNRESOLVED (turn stays open, mirroring a real CLI awaiting
                the tool result). turn 2: resolves turn-1's response with
                an end_turn stopReason and returns any second session/prompt
                the client sends with a fresh agent text + end_turn. Records
                every session/prompt payload to FAKE_ACP_TURN_ENDING_PROMPTS
                (one line per prompt: JSON of the prompt blocks) so the
                regression test can assert whether a second session/prompt
                was ever fired with the new HumanMessage payload.
  permission  — session/prompt first asks a permission request (Bash), then
                streams text and ends the turn.
  limit       — session/prompt replies with a JSON-RPC error whose text
                matches the claude usage-limit markers.
  crash       — session/prompt streams one chunk then the process exits
                non-zero (EOF on stdout).
  overflow    — session/prompt floods the notification sink past its
                capacity without ever ending the turn.

The permission reply (allow/deny) is echoed back on stdout prefixed with
``PERM:`` so the test can assert what the client chose.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

SCENARIO = os.environ.get("FAKE_ACP_SCENARIO", "text")


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def http_tools_call(url: str, name: str, args: dict, rid: int) -> dict:
    """Call the in-agent proxy MCP server (streamable HTTP) and return the
    JSON-RPC ``tools/call`` reply. Blocks until the handler is resolved by
    ``proxy_mcp.resolve_call`` (or its bounded wait elapses and it returns
    an in_progress job-handle instead)."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": rid,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
    ).encode()
    # Starlette Mount("/mcp") redirects POST "/mcp" → "/mcp/" (307); urllib
    # does not follow 307 for POST. Normalise to the trailing-slash form the
    # mount accepts directly (real CLIs follow the redirect themselves).
    if url and not url.endswith("/"):
        url = url + "/"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 — surface the failure to tests
        dbg_file = os.environ.get("FAKE_ACP_DEBUG_FILE")
        if dbg_file:
            with open(dbg_file, "a") as f:
                f.write(f"HTTP-ERROR {type(exc).__name__}: {exc}\n")
        return {"error": {"message": f"http_tools_call failed: {type(exc).__name__}: {exc}"}}
    if "text/event-stream" in ctype:
        for block in raw.split("\n\n"):
            payload = None
            for line in block.splitlines():
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
            if payload is None:
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                return obj
    try:
        return json.loads(raw)
    except ValueError:
        return {"error": {"message": f"unparseable proxy reply: {raw[:200]!r}"}}


def main() -> int:
    session_id: str | None = None
    proxy_url: str | None = None
    proxy_server_name: str | None = None
    # turn-ending-open bookkeeping: rid of turn-1's session/prompt so we can
    # resolve it when turn 2 arrives.
    turn_ending_pending_rid: int | None = None
    turn_ending_prompts_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        rid = obj.get("id")
        method = obj.get("method")
        params = obj.get("params") or {}

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "protocolVersion": 1,
                        "clientCapabilities": {},
                        "serverInfo": {"name": "fake-acp", "version": "0.0.1"},
                        "loadSession": True,
                    },
                }
            )
        elif method == "authenticate":
            send({"jsonrpc": "2.0", "id": rid, "result": {"authenticated": True}})
        elif method == "session/new":
            session_id = f"fake-{params.get('cwd', '.')}-{rid}"
            for srv in params.get("mcpServers") or []:
                if isinstance(srv, dict) and srv.get("type") == "http":
                    proxy_url = srv.get("url")
                    proxy_server_name = srv.get("name")
                    break
            dbg_file = os.environ.get("FAKE_ACP_DEBUG_FILE")
            if dbg_file:
                with open(dbg_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "mcpServers": params.get("mcpServers"),
                                "proxy_server_name": proxy_server_name,
                                "proxy_url": proxy_url,
                            }
                        )
                        + "\n"
                    )
            send(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"sessionId": session_id},
                }
            )
            # Give the client's reader a beat to process the reply and arm
            # the one-shot readiness future BEFORE the readiness notification
            # (a lost event would wedge wait_session_ready).
            time.sleep(0.05)
            # Readiness signal: first session-scoped notification.
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"text": ""},
                        },
                    },
                }
            )
        elif method == "session/load":
            session_id = params.get("sessionId")
            for srv in params.get("mcpServers") or []:
                if isinstance(srv, dict) and srv.get("type") == "http":
                    proxy_url = srv.get("url")
                    proxy_server_name = srv.get("name")
                    break
            send(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {"sessionId": session_id},
                }
            )
            time.sleep(0.05)
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"text": ""},
                        },
                    },
                }
            )
        elif method == "session/prompt":
            sid = params.get("sessionId")
            # Signed gate: record EVERY session/prompt request. The
            # Channel-A integration test asserts this file contains exactly
            # one line — a second prompt means the mid-turn resume wrongly
            # started a new turn.
            count_file = os.environ.get("FAKE_ACP_COUNT_FILE")
            if count_file:
                with open(count_file, "a") as f:
                    f.write("prompt\n")
            if SCENARIO == "limit":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "error": {
                            "code": -32000,
                            "message": "Request failed: You've hit your session "
                            "limit \u00b7 resets in 5 hours",
                        },
                    }
                )
                continue
            if SCENARIO == "overflow":
                # Flood the sink; never end the turn.
                for _ in range(10_000):
                    send(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": sid,
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {"text": "x" * 10},
                                },
                            },
                        }
                    )
                continue
            if SCENARIO == "permission":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": 9001,
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": sid,
                            # NOTE: no top-level "kind"; toolCall sits on
                            # params (mirrors real claude-agent-acp, where
                            # the tool name lives in
                            # toolCall._meta.claudeCode.toolName).
                            "toolCall": {
                                "toolCallId": "perm-1",
                                "title": "Bash",
                                "_meta": {"claudeCode": {"toolName": "Bash"}},
                            },
                            "options": [
                                {"kind": "allow", "optionId": "allow"},
                                {"kind": "reject", "optionId": "reject"},
                            ],
                        },
                    }
                )
            if SCENARIO == "turn-ending-open":
                # Regression harness for the mid-turn resume swallow
                # (rg1_worker, 2026-08-22, 2/2 live). Turn 1 emits a
                # turn-ending tool_call and leaves session/prompt
                # UNRESOLVED — a real CLI keeps the turn open while it
                # waits for the tool result.
                # Turn 2 resolves turn-1 (mirroring the CLI answering
                # once it eventually sees the tool result), and, if the
                # client actually sent a second session/prompt with the
                # NEW HumanMessage, answers that one too with a fresh
                # agent text + end_turn.
                turn_ending_prompts_count += 1
                prompts_log = os.environ.get("FAKE_ACP_TURN_ENDING_PROMPTS")
                if prompts_log:
                    blocks = params.get("prompt") or []
                    with open(prompts_log, "a") as f:
                        f.write(json.dumps(blocks) + "\n")

                tool_name = os.environ.get(
                    "FAKE_ACP_TOOL_NAME", "send_message_to_agent"
                )
                tool_args = json.loads(
                    os.environ.get(
                        "FAKE_ACP_TOOL_ARGS",
                        '{"agent_name":"peer","message":"delivered"}',
                    )
                )
                if turn_ending_prompts_count == 1:
                    # Emit the tool_call, remember rid, DO NOT resolve.
                    send(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "sessionId": sid,
                                "update": {
                                    "sessionUpdate": "tool_call",
                                    "toolCallId": "te-open-1",
                                    "title": tool_name,
                                    "rawInput": tool_args,
                                    "_meta": {
                                        "claudeCode": {
                                            "toolName": (
                                                f"mcp__{proxy_server_name}__"
                                                f"{tool_name}"
                                            )
                                        }
                                    },
                                },
                            },
                        }
                    )
                    turn_ending_pending_rid = rid
                    continue
                # Second (or later) session/prompt: this is the point of
                # the test. First resolve turn-1 so the client's still-open
                # future does not leak, THEN answer turn-2 with a fresh
                # agent text + end_turn.
                if turn_ending_pending_rid is not None:
                    send(
                        {
                            "jsonrpc": "2.0",
                            "id": turn_ending_pending_rid,
                            "result": {
                                "result": {"stopReason": "end_turn"}
                            },
                        }
                    )
                    turn_ending_pending_rid = None
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"text": "answer to the new q"},
                            },
                        },
                    }
                )
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "result": {"result": {"stopReason": "end_turn"}},
                    }
                )
                continue
            if SCENARIO == "tool":
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "t1",
                                "title": "Bash",
                                "rawInput": {"command": "ls"},
                                "_meta": {"claudeCode": {"toolName": "Bash"}},
                            },
                        },
                    }
                )
            if SCENARIO == "tool-then-end":
                # Regression harness for the _finalize_response tool_call
                # drop: emit a proxy-MCP tool_call and END THE TURN in the
                # same batch, without blocking on tools/call. Both the
                # notification and the prompt reply are written back-to-back,
                # so AcpChatModel sees the update queued AND response_fut
                # resolved on the same event loop iteration.
                tool_name = os.environ.get("FAKE_ACP_TOOL_NAME", "acp_probe_tool")
                tool_args = json.loads(
                    os.environ.get("FAKE_ACP_TOOL_ARGS", '{"question": "probe"}')
                )
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "t-same-tick-1",
                                "title": tool_name,
                                "rawInput": tool_args,
                                "_meta": {
                                    "claudeCode": {
                                        "toolName": (
                                            f"mcp__{proxy_server_name}__{tool_name}"
                                        )
                                    }
                                },
                            },
                        },
                    }
                )
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": rid,
                        "result": {
                            "result": {
                                "stopReason": "end_turn",
                                "_meta": {
                                    "inputTokens": 10,
                                    "outputTokens": 5,
                                    "totalTokens": 15,
                                },
                            }
                        },
                    }
                )
                continue
            if SCENARIO == "tool-proxy":
                # Channel A: a proxy MCP tool surfaced with the
                # claude-agent-acp naming scheme mcp__<server>__<tool>.
                tool_name = os.environ.get("FAKE_ACP_TOOL_NAME", "acp_probe_tool")
                tool_args = json.loads(
                    os.environ.get(
                        "FAKE_ACP_TOOL_ARGS", '{"question": "probe"}'
                    )
                )
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": sid,
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "t-proxy-1",
                                "title": tool_name,
                                "rawInput": tool_args,
                                "_meta": {
                                    "claudeCode": {
                                        "toolName": (
                                            f"mcp__{proxy_server_name}__"
                                            f"{tool_name}"
                                        )
                                    }
                                },
                            },
                        },
                    }
                )
                # Block on tools/call until the MCP handler is resolved by
                # proxy_mcp.resolve_call (driven by the LangGraph ToolMessage
                # path). If the handler instead returns an intermediate
                # in_progress job-handle (resolve never ran), that reply is
                # what lands in FAKE_ACP_RESULT_FILE and the test fails.
                reply = http_tools_call(
                    proxy_url, tool_name, tool_args, rid=7001
                )
                result_file = os.environ.get("FAKE_ACP_RESULT_FILE")
                if result_file:
                    with open(result_file, "w") as f:
                        f.write(json.dumps(reply))
            # Final answer chunk — tool-proxy uses an explicit marker so the
            # integration test can assert the post-tool text actually arrived.
            final_text = "hello from fake"
            if SCENARIO == "tool-proxy":
                final_text = "final from fake"
            send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": sid,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"text": final_text},
                        },
                    },
                }
            )
            if SCENARIO == "crash":
                # Die WITHOUT answering the prompt: the client's response
                # future must fail with an EOF error, not resolve.
                return 1
            send(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "result": {
                            "stopReason": "end_turn",
                            "_meta": {
                                "inputTokens": 10,
                                "outputTokens": 5,
                                "totalTokens": 15,
                            },
                        }
                    },
                }
            )
        elif method == "session/cancel":
            send({"jsonrpc": "2.0", "id": rid, "result": {"ok": True}})
        elif method == "session/request_permission":
            pass  # handled inside session/prompt above
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {"code": -32601, "message": f"{method} not supported"},
                }
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
