# Follow-up: `_PROVIDER_FALLBACK_HOURS` still keys cooldown by provider name

Status: OPEN, not scheduled. Deliberately out of scope for the configurable
failover chains work (`44267c6` docs, `42d2d51` implementation).
Agreed at sign-off: recorded here, not fixed there.

Reported 2026-09-03.

## What is left hardcoded

`runtime/availability_tracker.py:50`:

    _PROVIDER_FALLBACK_HOURS: dict[str, int] = {
        "claude-code-sdk": 6,
        "claude-code": 6,
        "codex": 6,
        "grok": 24,
        "minimax": 3,
    }

Read at `availability_tracker.py:103` via `.get(provider)`. This is the
recovery window used ONLY when the provider's own error message carries no
reset time — the cooldown that decides how long an exhausted provider stays
skipped in `~/.deep_worker/provider_availability.json`.

## Why it is worth revisiting

The 2026-09-03 owner directive was that failover chains must be configurable
and never hardcoded. That directive was executed for chain *composition*:
`PROVIDER_FAILOVER_CHAIN` is gone and chains live in
`~/.deep_worker/failover_chains.json`.

The cooldown table was not in scope and still is not configurable. It keys on
provider *name*, so it has the same shape as the defect that was just removed:

- A config-driven provider that is not spelled in this dict gets no fallback
  window at all. There are nine such providers today (`openrouter`, `minimax`,
  seven `or-*`), and only `minimax` is listed.
- Adding a provider through config alone cannot give it a cooldown. The source
  has to be edited, which is the thing the owner's directive is against.

Note this is a *narrower* problem than the chain hardcode was. An absent entry
does not strand an agent: it only means no synthetic cooldown when the provider
gave no reset time. The chain hardcode, by contrast, produced an empty
candidate list and left the agent in ERROR permanently.

## Do not "fix" `minimax: 3` by deleting it

The docstring above the table records that `3` is an owner directive from
2026-08-17, justified by measurement against the live API: the cap message is
`Token Plan usage limit reached … (2056)` with no time, every quota endpoint
candidate 404s, and there is no `x-ratelimit-*` / `retry-after` header. A
fallback is the only signal available and 3h is the operator's choice.

So the follow-up is to make the window *configurable* (and give the other
config-driven providers a way to declare one), preserving 3h for `minimax` as
the configured default — not to drop the number.

## Suggested shape (not designed, not signed off)

Let a provider declare its own fallback window in `providers.json`, next to
`account_group`, and keep this table only as the seed for the built-in
wrapper providers. That mirrors how `failover_chains.json` was seeded once
and then never consulted again. Requires design + review before any code.
