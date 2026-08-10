# DESIGN — LSP: auto-detection of languages and their management from the CLI

Status: draft for review
Branch: `dev`
Related: `DESIGN_LSP_SHARED_SERVICE.md`,
`DESIGN_LSP_SERENA_BACKEND_ATTACH.md`

## §1. Problem

In the target project (a Kotlin application + Python backend in `functions/`)
Serena refused to work with Python:
`Cannot extract symbols. Active languages: ['kotlin']`.

The cause was established by reading Serena's sources. Auto-detection on first
activation enables **exactly one language — the most frequent one**
(`serena/config/serena_config.py:376-382`):

```python
languages_and_percentages = sorted(...)      # by number of files
top_language_pair = languages_and_percentages[0]
languages_to_use = [top_language_pair[0]]    # ← only the first
```

The rest are added **only in interactive mode** (line 384,
`if len(other_language_pairs) > 0 and interactive`). We run Serena as an
MCP server — there is no interactivity, no one asks the question, and the
list is silently truncated to one language.

The actual composition of the target project (computed by Serena itself,
respecting `.gitignore`):

| Language | Share |
|---|---|
| kotlin | 65.42 % |
| **python** | **31.83 %** |
| bash | 2.68 % |
| typescript / vue / svelte | 0.07 % each |

Python is a third of the codebase, and it silently dropped out.

Why this needs fixing on our side rather than being dismissed as a third-party
quirk:

1. The agent can neither see the problem (it does not look at the config) nor
   fix it (there is no tool for that). It only sees the tool failure and
   builds false theories — exactly as in the `-32800` case.
2. The same trap will recur on any polyglot project.
3. Serena is launched by us, and the config lives in the project's working
   directory — the responsibility is ours.

### User requirements

- Language auto-detection **by default**, respecting `.gitignore`,
  threshold **10 %**.
- `deep-worker lsp` commands for explicitly adding/removing a language.

## §2. What is NOT in scope

- We do not patch Serena or solidlsp — we only read their public functions.
- We do not change `serena_config.yml` (the user's global config) — we only
  touch `<project>/.serena/project.yml`.
- We do not change gRPC/proto, the bridge, the watcher, the lock, or retry.
- We add no threads.

## §3. Key decision: use Serena's own detector

`serena.util.inspection.determine_programming_language_composition(repo)`
already does exactly what is needed:

- it walks files via `find_all_non_ignored_files` — **`.gitignore` is
  respected**, the requirement is satisfied "for free";
- it maps extensions with the official language matchers;
- it returns `dict[Language, float]` — percentages of **recognized**
  sources (images/locales do not dilute).

We will not write our own detector: it would inevitably diverge from Serena's
language list and its own ignore rules. This is exactly the case where
duplicating third-party logic is worse than depending on it.

The import is **lazy and fail-quiet**: `deep-worker` must work even without
Serena installed (it already does — the CLI checks this). No Serena → no
auto-detection, and the command explains why.

## §4. Auto-detection (10 % threshold)

### §4.1. When it fires

At `deep-worker lsp enable` — the moment when the user explicitly enables
navigation for a project. This is the only point where we already write
configs and where a delay for walking the files is appropriate.

We do **not** touch the service autostart: it is on the hot path of every
agent launch, and walking the tree costs time.

**Amendment A4 (a repeated `enable` must reconcile languages).** Today,
`enable` on an already-enabled project is a no-op: `enable_lsp_at_path`
sees an existing MCP entry and does nothing. If detection were tied to the
fact that `mcp.json` changed, then for **already-installed** projects (and
this is exactly the target-project case) auto-detection would never fire —
i.e. the "by default" requirement would not be met.

Therefore: the no-op applies **only to the MCP entry**; language
reconciliation (detection + merge) runs on every `enable` independently of
it.

Failure handling differs by the intent of the call:

| Situation | Behavior |
|---|---|
| `enable`, detection failed / no Serena | fail-quiet: a warning; the `enable` result is preserved (languages are not the command's main goal) |
| explicit `languages --detect`, detection failed | **exit 1** with the reason: the user asked for exactly this, so "success with no changes" would be a lie |

### §4.2. Logic

```
composition = determine_programming_language_composition(root)   # {lang: pct}
detected    = [lang for lang, pct in composition if pct >= 10.0]
```

The threshold `_LANGUAGE_THRESHOLD_PCT = 10.0` comes from the requirement.

Merging with the existing config:

- no `project.yml` file → we create a **valid** config from Serena's
  official template (see A1 below);
- file exists → we **merge**: `languages = existing ∪ detected`,
  preserving order (existing first — Serena treats the first language as
  the default one, and it must not change);
- nothing changed → the file is not rewritten (we do not touch mtime).

**Amendment A1 (creating a new config).** The original plan "write a minimal
file with only `languages:` and let Serena fill in the rest" is **wrong** —
verified against the sources:
`ProjectConfig.FIELDS_WITHOUT_DEFAULTS = {"project_name", "languages"}`
(`serena_config.py:341`). Fields in this set **do not receive default
values** on load, so a file without `project_name` is invalid, and Serena
will not "fill it in".

Correct: generate from the official template
`serena/resources/project.template.yml` (the constant
`PROJECT_TEMPLATE_FILE`), substituting `project_name = <the project
directory's name>` and the computed `languages`. The template carries the
whole comment block and all other keys with defaults.

The test must **actually load** the created file via `ProjectConfig`, not
just use `yaml.safe_load`: only loading proves the config is valid for
Serena.

**An important detail found while checking A1:** in the template,
`languages` is written **inline** — `languages: ["python"]` (line 36),
not as a block list. So the editor from §4.4 must handle both formats or
explicitly refuse — see A2.

**Languages are never removed automatically.** The user may have added a
language by hand for a specific task; silently dropping it because it did
not reach 10 % would lose a user decision. Removal is only via an explicit
command (§5).

### §4.3. Permissions

`project.yml` in the target project is currently `0600`. On write:

- existing file — **mode preserved as is** (we do not widen permissions);
- new file — `0600`, the `.serena/` directory — `0700`.

The "no 777/666" policy is respected.

### §4.4. Preserving comments

`project.yml` is a Serena template with a large comment block (the list of
supported languages, etc.). A blunt rewrite via `yaml.safe_dump` would
destroy them.

Solution: **a targeted edit of the `languages:` block** — we read the file
line by line, replace only the list items under the `languages:` key, and
return everything else byte-for-byte. This is more robust than full parsing
and does not pull in a dependency on `ruamel.yaml`.

### Amendment A2 — contract of the safe editor

A targeted text edit is parsing a third-party format, and it must be
**strict and fail-loud**, not "guessy". The contract:

1. **Semantic validation first.** Before mutating, the file is parsed with
   `yaml.safe_load` — to make sure the document is valid at all and that
   `languages` exists exactly once at the top level. Parsing is used ONLY
   for validation; we write the original text with a targeted replacement.
2. **Two formats are supported** (both actually occur):
   - block list (`languages:\n- kotlin`) — as in the target project;
   - inline (`languages: ["python"]`) — as in Serena's template (A1).
   Both are covered by tests.
3. **Refusal instead of guessing.** Fail-loud with a clear error on: a
   duplicate top-level `languages` key; a match that is only nested or
   commented `languages:`; a missing key in an existing file; invalid YAML;
   anchors/tags/multiline around the block. The file is **not touched** in
   these cases.
4. **Byte-for-byte outside the block.** All text outside the replaced
   block, the line-ending style, and the presence/absence of a trailing
   `\n` are preserved.

### Amendment A3 — atomic writes

Write via a temporary file in the **same directory** + `fsync` +
`os.replace` (the same pattern as in `ServiceRegistry.write`). The temporary
file is created with `0600`.

- existing file — we restore the **exact** original mode;
- new file — `0600`, the `.serena/` directory — `0700`.

On any parse or write error the original stays **byte-identical**: partial
writes are impossible. Covered by a test.

## §5. CLI

```
deep-worker lsp languages [--project-root DIR]           # show
deep-worker lsp languages --add python [--add go]        # add
deep-worker lsp languages --remove kotlin                # remove
deep-worker lsp languages --detect                       # re-run auto-detection
```

The `languages` subcommand with flags was chosen over `lsp --add python`:
in the existing CLI, `lsp` is a group of subcommands
(`enable`/`disable`/`status`/`serve`/…), and a flag on the group itself
would break the pattern. The `--add/--remove` form is exactly what the user
asked for.

Behavior:

- `--add`/`--remove` accept a language name; **validation against
  `Language.iter_all()`** (48 languages). An unknown name → an error with a
  list of similar ones, not a silent write of garbage into the config.
- `--remove` of the last language → refusal: an empty list means "symbol
  tools do not work", which is almost certainly not what is wanted. We
  explain explicitly and suggest `disable`.
- Without flags — prints the current languages, the detected composition,
  and marks which languages would pass the threshold but are not enabled.
  This is the diagnostics for the target-project case.
- Changes require a service restart — we print a hint
  (`deep-worker lsp stop`), we do not restart ourselves: a restart would
  interrupt other agents working on this project.

Exit codes: `0` — success/no change, `1` — error (no Serena, unknown
language, attempt to remove the last one).

### Amendment A5 — semantics of combined flags

The order of operations is fixed and documented, with no hidden
priorities:

1. `--detect` (if given) → merge the detected;
2. all `--add` in the order they appear on the command line;
3. all `--remove` in the order they appear.

Rules:

- the same language in both `--add` and `--remove` → **refused as a
  conflict** (exit 1). A silent "last one wins" would hide a user typo;
- repeated flags are de-duplicated;
- `--add` of an already-present language and `--remove` of an absent one —
  idempotent no-op, exit 0;
- the final list is validated **before** writing; an empty result is
  forbidden (see §5);
- **without flags the command is strictly read-only**: it does not create
  or change `project.yml`. If the file does not exist, it prints the
  detected composition and an explicit "not configured", mutating nothing.

Percentages in the output are labeled **"% of recognized source files"** —
the denominator is exactly that (§4), and without the label the number
would mislead.

### Amendment A6 — order of detected languages

Detected languages are sorted **by the same rule Serena uses**:
percentage descending, ties broken by `get_priority()` descending
(`serena_config.py:376-378`). Otherwise the order would depend on dict
iteration and could accidentally assign the default language (for Serena
this is the first in the list).

- existing languages always come first and are **not reordered**;
- detected ones are only appended at the end; no auto-removal.

Empty composition (no recognized sources):

| Call | Behavior |
|---|---|
| `enable` | warning; the config is **not** mutated |
| `languages --detect` | exit 1 with an explicit "no supported sources found" |

### Amendment A7 — restart hint

After a **real** change, we print that it will take effect after stopping
the current shared service, plus the exact safe command with the same
`--project-root`. We do not restart ourselves: that would interrupt other
agents working on the project.

On a no-op the hint is **not** printed — otherwise the user would restart
the service without reason.

## §6. Delta

| File | Change |
|---|---|
| `lsp/languages.py` | **NEW** — detection, merge, read/write of `project.yml` |
| `lsp/enable.py` | call auto-detection on `enable` |
| `cli/lsp_command.py` | the `languages` subcommand + handler |

Unchanged: the service, backend, bridge, proto, watcher, retry.

## §7. Tests (TDD)

### Unit `test_lsp_languages.py` (new)

1. `test_detect_applies_ten_percent_threshold` — composition
   `{kotlin: 65, python: 32, bash: 2.7}` → `[kotlin, python]`,
   bash dropped.
2. `test_detect_respects_gitignore` — the detector is run on a real
   temporary repository with a `.gitignore`; ignored files do not affect
   the result (verifies that we use Serena's function, not our own walk).
3. `test_merge_keeps_existing_and_adds_detected` — the config has
   `[kotlin]`, detected `[kotlin, python]` → `[kotlin, python]`,
   order preserved.
4. `test_merge_never_removes_configured_language` — the config has
   `[kotlin, go]`, detected `[kotlin]` → `go` stays.
5. `test_write_preserves_comments_and_other_keys` — after the edit,
   `project_name`, comments, and other keys are **byte-for-byte** intact.
6. `test_write_preserves_file_mode` — a `0600` file stays `0600`.
7. `test_new_config_is_created_0600_in_0700_dir`.
8. `test_no_write_when_nothing_changes` — mtime does not change.
9. `test_add_validates_language_name` — `--add nosuch` → error,
   config untouched.
10. `test_remove_last_language_refused` — refusal + clear text.
11. `test_add_is_idempotent` — a repeated `--add python` does not duplicate.

**A1 (validity of a new config):**

12. `test_new_config_loads_through_serena_project_config` — the created
    file is actually loaded by `ProjectConfig`, not only by
    `yaml.safe_load`; `project_name` and `languages` are in place.

**A2 (strict editor):**

13. `test_inline_languages_list_is_supported` — Serena's template format
    `languages: ["python"]` is edited correctly.
14. `test_duplicate_top_level_key_refused` — file untouched.
15. `test_nested_or_commented_languages_not_matched` — a `languages:`
    inside a nested mapping/in a comment is not treated as a target;
    refusal, file untouched.
16. `test_malformed_yaml_refused` — refusal before any write.
17. `test_missing_key_in_existing_file_refused`.
18. `test_newline_style_and_trailing_newline_preserved` — CRLF and a
    missing trailing `\n` are preserved.

**A3 (atomicity):**

19. `test_write_failure_leaves_original_byte_identical` — a write failure
    (mocked `os.replace`) → the original is unchanged, the temp file is
    removed.

**A4 (repeated enable):**

20. `test_repeat_enable_reconciles_languages` — the project is already
    enabled, `mcp.json` is not rewritten, but Python is appended to
    `project.yml`.
21. `test_enable_survives_detection_failure` — detection throws →
    `enable` returns its previous success + a warning.
22. `test_explicit_detect_fails_loud` — with Serena unavailable,
    `languages --detect` exits 1.

**A5/A6 (CLI semantics and order):**

23. `test_add_and_remove_same_language_conflicts` — exit 1.
24. `test_flag_order_detect_then_add_then_remove`.
25. `test_no_flags_is_read_only` — `project.yml` is neither created nor
    changed (checked via mtime and existence).
26. `test_detected_sorted_by_pct_then_priority` — the order is
    deterministic, independent of dict key order.
27. `test_empty_composition_contract` — `enable` does not mutate,
    `--detect` exits 1.

### Unit `test_lsp_command.py` (addition)

28. `test_languages_subcommand_registered` — the parser accepts
    `languages --add python`.

### Integ `test_lsp_languages_e2e.py` (new, no mocks)

29. A real temporary polyglot project (Kotlin + Python + junk cut off by a
    `.gitignore`, **including a nested `.gitignore`** — exactly what the
    parser in use promises) → `lsp languages --detect` through the **real
    CLI** → both languages in `project.yml`, permissions `0600`.
30. `lsp languages` (no flags) prints the composition with the
    "% of recognized source files" label and the current list.
31. `test_repeat_enable_adds_missing_language_e2e` — the target-project
    scenario end to end: a project already enabled with one language, a
    repeated `lsp enable` appends the second.

## §8. Mutations

- **MUT-L1**: threshold `>= 10` → `> 0` — test 1 fails (bash would get in).
- **MUT-L2**: the merge replaces instead of unioning — test 4 fails.
- **MUT-L3**: write via a full `yaml.dump` — test 5 fails
  (comments would disappear).
- **MUT-L4**: `chmod 0644` on creation — test 7 fails.
- **MUT-L5**: remove the language-name validation — test 9 fails.
- **MUT-L6**: allow removing the last language — test 10 fails.
- **MUT-L7** (A1): create a config without `project_name` — test 12
  fails (Serena would not load it).
- **MUT-L8** (A2): on an ambiguous match, edit the first occurrence
  instead of refusing — tests 14/15 fail.
- **MUT-L9** (A3): write directly to the file without temp+replace — test
  19 fails (a partial write remains).
- **MUT-L10** (A4): run detection only when `mcp.json` changes — test 20
  fails (the target-project scenario is not fixed).
- **MUT-L11** (A5): allow add+remove of the same language with a hidden
  priority — test 23 fails.
- **MUT-L12** (A6): sort detected languages by dict key order — test 26
  fails.

## §9. Open questions

None. The threshold, `.gitignore` respect, and the command form were set
explicitly by the user; the rest are consequences.

## §10. Compatibility

- Projects with an already-configured `project.yml` only get an **addition**
  to the list; existing languages and their order are preserved.
- No Serena → auto-detection is skipped, `enable` works as before.
- Configs of other projects are not affected (we only edit the selected
  `--project-root`).
