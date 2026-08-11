"""DEPRECATED shim — migrate to ``deep_worker.providers.grok_provider``.

This root package exists only for local backward compatibility on this host.
It is NOT part of the published wheel/sdist. Old submodule imports
(``import grok_provider.chat_model``) are NOT supported: use
``deep_worker.providers.grok_provider.chat_model`` instead.
"""
import warnings as _warnings

_warned = False
__all__ = ["GrokChatModel"]


def __getattr__(name):
    global _warned
    if not _warned:
        _warned = True
        _warnings.warn(
            "grok_provider is deprecated; use "
            "deep_worker.providers.grok_provider",
            DeprecationWarning,
            stacklevel=2,
        )
    from deep_worker.providers import grok_provider as _canonical

    return getattr(_canonical, name)
