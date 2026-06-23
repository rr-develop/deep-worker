# Provider failover chain constant.
#
# Order of automatic provider failover on usage-limit errors (issue #77).
# Each failover advances one step down the list; ``deepseek`` is terminal.
#
#   claude-code → codex → grok → deepseek
#
# Passthrough built-ins and MCP tools are unchanged by failover — only the
# active ``GrokChatModel`` / ``CodexChatModel`` / etc. swap on restart.
PROVIDER_FAILOVER_CHAIN: list[str] = ["claude-code", "codex", "grok", "deepseek"]
