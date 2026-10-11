"""Context-limit guarantee (rg1_worker 2026-08-18).

See ``docs/DESIGN_CONTEXT_LIMIT_GUARANTEE.md``.

This package provides:

* ``observed_context`` — the PROVIDER-reported context size, the only
  number the guard and the summarizer may decide on (owner directive
  2026-08-25). There is deliberately no estimator: the chars-per-token
  table that used to live here reported 214613 against a real 90k and
  drove both a false summarization and a dropped provider session.
* ``guard`` — the ``ContextLimitGuardMiddleware`` that sits ahead of
  ``SummarizationMiddleware`` in the agent chain and enforces the
  three-band policy (Green / Amber / Red / Block).
* ``provider_max`` — the ``PROVIDER_MODEL_MAX_INPUT`` table that
  surfaces provider-imposed ceilings at boot time.
"""

from __future__ import annotations