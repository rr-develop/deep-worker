"""DEPRECATED shim — migrate to ``deep_worker.providers.codex_provider``.

This root package exists only for local backward compatibility on this host.
It is NOT part of the published wheel/sdist. Old submodule imports
(``import codex_provider.native_tools``) are NOT supported: use
``deep_worker.providers.codex_provider.native_tools`` instead.
"""
import warnings as _warnings

_warned = False
__all__ = ["CodexChatModel"]


def __getattr__(name):
    global _warned
    if not _warned:
        _warned = True
        _warnings.warn(
            "codex_provider is deprecated; use "
            "deep_worker.providers.codex_provider",
            DeprecationWarning,
            stacklevel=2,
        )
    from deep_worker.providers import codex_provider as _canonical

    return getattr(_canonical, name)
