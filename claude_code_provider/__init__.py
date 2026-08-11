"""DEPRECATED shim — migrate to ``deep_worker.providers.claude_code_provider``.

This root package exists only for local backward compatibility on this host.
It is NOT part of the published wheel/sdist. Old submodule imports
(``import claude_code_provider.web_search``) are NOT supported: use
``deep_worker.providers.claude_code_provider.web_search`` instead.
"""
import warnings as _warnings

_warned = False
__all__ = [
    "ClaudeCodeChatModel",
    "ClientManager",
    "ProxyMCPServer",
    "WebSearchDelegationTool",
]


def __getattr__(name):
    global _warned
    if not _warned:
        _warned = True
        _warnings.warn(
            "claude_code_provider is deprecated; use "
            "deep_worker.providers.claude_code_provider",
            DeprecationWarning,
            stacklevel=2,
        )
    from deep_worker.providers import claude_code_provider as _canonical

    return getattr(_canonical, name)
