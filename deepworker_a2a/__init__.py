"""DEPRECATED shim — migrate to ``deep_worker.providers.deepworker_a2a``.

This root package exists only for local backward compatibility on this host.
It is NOT part of the published wheel/sdist. Old submodule imports
(``import deepworker_a2a.config``) are NOT supported: use
``deep_worker.providers.deepworker_a2a.config`` instead.
"""
import warnings as _warnings

_warned = False
__all__ = ["A2AConfig"]


def __getattr__(name):
    global _warned
    if not _warned:
        _warned = True
        _warnings.warn(
            "deepworker_a2a is deprecated; use "
            "deep_worker.providers.deepworker_a2a",
            DeprecationWarning,
            stacklevel=2,
        )
    from deep_worker.providers import deepworker_a2a as _canonical

    return getattr(_canonical, name)
