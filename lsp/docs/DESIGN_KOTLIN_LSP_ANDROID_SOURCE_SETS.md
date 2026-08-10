# DESIGN: cross-file navigation for the Kotlin LSP on Android projects (AGP)

Status: **DESIGN SIGNED OFF** (2026-07-31) by the review lead.
Branch `dev`. Implementation approved per the TDD plan in §10.
The reviewer's answers to the open questions are incorporated in §13.

Related documents: `lsp/docs/DESIGN_LSP_MCP_INTEGRATION.md` (baseline
Serena integration), `lsp/README.md` (user documentation).

Follow-ups are moved to §11 and are **out of scope** for this design.

---

## 1. Symptom (observed, reproduced many times)

On an Android project (SampleApp, AGP 8.9.1, Kotlin 2.1.0, Gradle 8.11.1)
navigation **within a single file** works, while **any cross-file operation**
silently returns empty results:

| Tool | Expected | Actual (before the fix) |
|---|---|---|
| `find_referencing_symbols` (cross-file) | list of usages | `{}` — silently, no error |
| `find_declaration` (cross-file) | declaration site | `ValueError: No symbol declaration found at the location of the regex match` |
| `find_implementations` | interface implementations | `[]` (LS returned `None`) |
| `workspace/symbol` | project symbols | `[]` |

At the same time `get_symbols_overview`, `find_symbol`, same-file
`find_declaration` and `get_diagnostics_for_file` work normally. This is what
misleads: the LSP looks "alive".

The results are **byte-for-byte identical** after a 70 s and a 210 s warm-up —
so it is not unfinished indexing; the import completed incorrectly and
repeating it is pointless.

## 2. First clue in the Kotlin LSP log

```
INFO - #c.j.l.i.g.IdeaProjectMapper - SampleApp has an empty set of source sets
INFO - #c.j.l.i.g.IdeaProjectMapper - SampleApp.app has an empty set of source sets
```

Checking the built workspace model confirms: **not a single source root**.

```
$ strings ~/.config/JetBrains/analyzer/workspaces/<hash>/workspace-model.cache     | grep 'WORKSPACE_0'
file://$WORKSPACE_0$/ap                                    # content root of the app module
jar://$WORKSPACE_0$/app/build/.../processDebugResources/R.jar!/
file://$WORKSPACE_0                                        # content root of the root project
```

Three entries — and `app/src/main/java` is not among them. At the same time
**302 libraries were imported successfully**:

```
INFO - #c.i.w.i.i.l.l.ProjectLibraryTableBridgeImpl - Initial load of
       project-level (project=analyzer project) libraries.
       There are 302 libraries to load.
```

The classpath is there, the sources are not → the project symbol index is
empty → cross-file resolution is impossible. This fully explains the table in
§1, including why same-file operations keep working (they go through the PSI
of the open document, not through the project index).

## 3. What was ruled out (verified, not assumed)

Before fixing, three plausible hypotheses were discarded.

**3.1. "Old LSP version, needs an upgrade".** The v262.4739.0 release notes
claim "Import of Android projects is now supported by Kotlin LSP". Installed
was v262.2310.0. The hypothesis looked primary. Upgraded to **v262.9593.0**
(latest, 27 Jul 2026, bundled JBR 25) — the symptom reproduced **unchanged**.
Hypothesis disproved; the upgrade was kept anyway (see §8).

**3.2. "Not enough time for the Gradle sync".** An upstream issue notes that
JetBrains LSP "legitimately needs 2-5 minutes on large projects". However the
log shows the import **completing successfully**:

```
INFO - [IMPORT STD]: > Task :app:prepareKotlinIdeaImport
INFO - [IMPORT STD]: BUILD SUCCESSFUL in 44s
INFO - [IMPORT STD]: 71 actionable tasks: 33 executed, 3 from cache, 35 up-to-date
```

The sync did not "run out of time" — it ran to completion and produced an
empty result.

**3.3. "The project's `org.gradle.configuration-cache=true` is to blame".**
The hypothesis was checked with a direct experiment: `configuration-cache=false`
in `gradle.properties` → the error is **the same**. Disproved; the setting was
restored.

## 4. Root cause: two independent Kotlin LSP defects

Both were established from the server bytecode (`javap`) and **confirmed by a
live experiment**, not guessed.

### 4.1. Defect 1 — Android variants are never collected

`AndroidVariants.collectAndroidVariants(Gradle)` registers a collector via
`gradle.getLifecycle().beforeProject(IsolatedAction)`, and inside the action
immediately looks up the extension:

```
// AndroidVariants.collectAndroidVariants(Gradle)
21: invokeinterface  org/gradle/api/invocation/Gradle.getLifecycle()
33: invokeinterface  GradleLifecycle.beforeProject(IsolatedAction)

// AndroidVariants.collectAndroidVariants(Project)  ← action body
13: invokestatic     setAndroidVariants(Project, List)        // puts an EMPTY mutable list
17: invokestatic     AndroidReflectionKt.getAndroidComponents(Project)
21: ifnull   36                                               // null → exit immediately
30: invokevirtual    AndroidComponentsExtensionReflection.onVariants(Function1)
```

And `getAndroidComponents` is simply:

```
12: ldc              "androidComponents"
14: invokeinterface  ExtensionContainer.findByName(String)
```

**The problem:** `beforeProject` fires **before** the project's plugins are
applied, so the extension does not exist yet. Experiment (an init script on
the same project):

```
BP-PROBE: [beforeProject]  :     androidComponents=NULL
BP-PROBE: [beforeProject]  :app  androidComponents=NULL
BP-PROBE: [afterEvaluate]  :     androidComponents=NULL
BP-PROBE: [afterEvaluate]  :app  androidComponents=PRESENT   ← it exists here
```

The server itself honestly writes this to the log (confirmation, not
indirection):

```
WARN - [IMPORT ERR]: Failed to call 'onVariants' in 'androidComponents' extension
WARN - [IMPORT ERR]: Failed to call 'onVariants' in 'androidComponents' extension
```

Result: `onVariants` is never registered → the variant list stays empty.

### 4.2. Defect 2 — source dirs cannot be read during the configuration phase

Even with the variants collected, resolution fails.
`AndroidSourceSets.resolveToModuleSourceSet` reads the variant's sources:

```
383: invokevirtual  AndroidComponentReflection.getSources()
390: invokevirtual  AndroidSourcesReflection.getKotlin()   → Provider<Collection<Directory>>
541: invokevirtual  AndroidSourcesReflection.getJava()     → Provider<Collection<Directory>>
501: invokeinterface org/gradle/api/file/Directory.getAsFile()   ← the Provider is evaluated here
```

AGP mixes **generated** directories into these Providers, mapped onto the
output of the `generate<Variant>BuildConfig` task. Gradle forbids reading such
a Provider before the producing task has run:

```
[probe] :app: resolveAndroidSourceSets FAILED:
org.gradle.api.InvalidUserCodeException: Querying the mapped value of
property(... flatmap(provider(task 'generateDebugBuildConfig',
class com.android.build.gradle.tasks.GenerateBuildConfig)) check-type()))))
before task ':app:generateDebugBuildConfig' has completed is not supported
    at org.gradle.api.internal.provider.TransformBackedProvider.lambda$beforeRead$0(TransformBackedProvider.java:98)
    at org.gradle.api.internal.provider.TransformBackedProvider.beforeRead(TransformBackedProvider.java:95)
    at org.gradle.api.internal.provider.TransformBackedProvider.calculateOwnValue(TransformBackedProvider.java:80)
```

The exception aborts the whole source-set resolution → `ProjectMetadata.getSourceSets()`
is empty for the module → the log from §2 fires.

The same exception (in its `generateDebugAndroidTestBuildConfig` variant) is
also visible in a **stock** import, without any of my scripts — i.e. the
defect is original, not introduced:

```
WARN - [IMPORT ERR]: Querying the mapped value of property(... task
'generateDebugAndroidTestBuildConfig' ...) before task ... has completed
is not supported
```

The prohibition condition in Gradle is exactly `task.state.executed`:

```
// TransformBackedProvider.beforeRead → ValueSupplier$TaskProducer
 1: invokeinterface  org/gradle/api/Task.getState()
 6: invokeinterface  org/gradle/api/tasks/TaskState.getExecuted()
18: ldc  "Querying the mapped value of %s before %s has completed is not supported"
```

and `getExecuted()` in `TaskStateInternal` is just "the `outcome` field is not
null":

```
public boolean getExecuted();
   0: getfield  outcome:Lorg/gradle/api/internal/tasks/TaskExecutionOutcome;
   4: ifnull    11
   7: iconst_1        // executed = (outcome != null)
```

**Important:** defect 2 is not specific to SampleApp. `buildConfig` is enabled
in AGP by default, and the project's `build.gradle.kts` contains nothing
exotic in this area — it reproduces on any AGP project with BuildConfig.

### 4.3. Why exactly two defects, not one

They are independent and both blocking. Verified step by step:

| State | `variants` | `resolveAndroidSourceSets` | source roots |
|---|---|---|---|
| as is | 0 (`onVariants` not called) | does not get there | 3 |
| only defect 1 fixed | 4 | `InvalidUserCodeException` | 3 |
| both fixed | 4 | success | **21** |

## 5. Proposed workaround

### 5.1. Defect 1 — rerun the collector at the right moment

`setAndroidVariants(project, list)` puts a **mutable** list into the project's
`Extras`, and the `onVariants` callback appends to the **same** object:

```
// collectAndroidVariants$lambda$0(List, AndroidVariantReflection)
 8: invokeinterface  java/util/List.add(Object)
```

So it is enough to call the collector again once the extension already exists
but the variants are not yet finalized — that is
`pluginManager.withPlugin("com.android.*")`. The list the importer will read
later turns out populated. Nothing in the server needs patching.

Result in the log: `[probe] :app: variants=4`.

### 5.2. Defect 2 — a narrow window around model building

`ModuleSourceSets` is built by a public `ToolingModelBuilder` registered in
`ToolingModelBuilderRegistry`. We register **our own wrapper** for the same
model; Gradle gives priority to the last registered builder. The wrapper:

1. marks `generate*BuildConfig` tasks as executed
   (`state.setOutcome(UP_TO_DATE)` — exactly the field read by the check from §4.2);
2. delegates building to the original builder;
3. **in `finally`** restores the previous `outcome`.

The mutation window is a single `buildAll` call, synchronous, with no
concurrent phases.

### 5.3. Why `setOutcome` and not disabling the check

The alternative "remove the prohibition globally" would require patching
Gradle or running it with an internal flag. The check reads a single field;
putting it back is a minimal and fully reversible intervention.

## 6. Risks

### 6.1. Mutating `task.state` — the main risk, analyzed separately

`getExecuted()` affects more than just Provider reads. In `TaskStateInternal`:

```
public boolean isConfigurable();
   0: invokevirtual  getExecuted()      // executed → NOT configurable
   8: getfield       executing
```

That is, while a task is marked executed it is considered non-configurable.
**This is not theory — I stepped on it.** An early version of the workaround
marked tasks in `afterEvaluate` (a wide window) and broke configuration:

```
[probe] :app: resolveAndroidSourceSets FAILED:
java.lang.IllegalStateException: Cannot call Task.finalizedBy(Object...)
on task ':app:generateDebugBuildConfig' after task has started execution.
```

That is exactly why the window is narrowed to a single `buildAll` and is
always closed in `finally`. In the final configuration this error is absent
from the log.

**Residual risk:** if the original `buildAll` throws, `finally` still restores
the state. If the process dies hard — the state lives only in the import
daemon's memory and is not persisted.

**Assertion risk.** `setOutcome` contains `assert outcome == null`:

```
public void setOutcome(TaskExecutionOutcome);
   0: getstatic  $assertionsDisabled
   3: ifne  21
   7: getfield  outcome
  10: ifnull    21
  13: new  java/lang/AssertionError
```

When the Gradle daemon runs with `-ea`, restoring `UP_TO_DATE → null` is not
affected (the assert only checks the transition from non-null), but a
**second** non-null-over-non-null assignment will fail. The wrapper marks only
tasks with `!state.executed` (i.e. `outcome == null`), so the assert
condition holds. This must be covered by a test (§10, T5).

**Blast radius** is limited: only tasks whose name starts with `generate` and
contains `BuildConfig` are marked, and only in projects that apply
`com.android.application`/`com.android.library`.

### 6.2. Fragility of the binding to internals

The workaround relies on private names (`access$collectAndroidVariants`,
`TaskExecutionOutcome`, the model name
`com.jetbrains.ls.imports.gradle.model.ModuleSourceSets`). Renaming them in a
new Kotlin LSP version will break the workaround.

Mitigation: **every step is wrapped in `try/catch`; any error → warning and
continue**. The worst outcome on a version mismatch is a return to the current
(broken) behavior, not an import crash. This is a mandatory implementation
requirement, covered by test T4.

### 6.3. Globality of `~/.gradle/init.d/`

The script applies to **all** of the user's Gradle builds, not only to the LSP
import. Mitigation: first thing, it checks for the `IdeaGradleLspPlugin`
plugin in `gradle.plugins`; outside an import it is an immediate no-op.
Verified: `./gradlew help` on SampleApp prints no markers and does not change
behavior. Covered by test T3.

### 6.4. What does NOT break

- Regular builds (`assemble`, `test`) — no-op per §6.3.
- Non-Android Gradle projects — no-op (no `com.android.*` plugin).
- Non-Gradle projects and other languages — the file is not even read.

## 7. Choosing where the script lives (rationale)

The task statement proposed three options. The key constraints that narrow the
choice (both verified, not taken on faith):

**Constraint A. Gradle does not read init scripts from the project.** Only
`$GRADLE_USER_HOME/init.d/*.gradle` are applied automatically. Verified by
experiment: a script in `<project>/gradle/` is not applied
(`NOT auto-applied from gradle/ dir`). So "put the file into the target
project's repository" physically does not work — you need either
`~/.gradle/init.d/` or an explicit `--init-script`, which we do not control in
the Kotlin LSP command line (the server builds it itself).

**Constraint B. `.serena/project.yml` is the target project's file, not ours.**
Serena creates it at the root of the checked project (`SampleApp/.serena/`);
it is not in our git and cannot be. Moreover, `deep_worker_cli` currently does
**not manage** either `project.yml` or `serena_config.yml` at all — a code
search yields no occurrences. Our code only writes `mcp.json`.

Hence the option evaluation:

| Option | Verdict |
|---|---|
| **(a)** resource in the package + installed at `lsp enable` | **Selected.** The only one that delivers "works for anyone who clones the repository": the file is versioned in git, installed by a standard command, and ends up in the single place Gradle will read it from. |
| **(b)** generate from `serena_config`/`project.yml` | Rejected. It would require us to start owning someone else's configs (constraint B) — an expansion of the area of responsibility and a conflict risk with Serena itself, which overwrites these files. Disablability is achieved more simply — via a CLI flag (§7.1). |
| **(c)** a documented step in the README | Rejected as a standalone solution: manual copying is not reproducible and silently degrades. **But accepted as a supplement** — the README must describe exactly what is installed, where, and how to remove it. |

### 7.1. Contract of option (a)

- Resource: `deep_worker_cli/lsp/assets/kotlin-android-source-sets.init.gradle`,
  packaged via the already-used `[tool.hatch.build.targets.wheel.shared-data]`.
- Installation: a **copy**, not a symlink. A symlink would point inside the
  venv/wheel and break on package reinstall; a copy is self-contained.
- Install point: `deep-worker lsp enable`, **enabled** by default; disabling
  is an explicit `--no-gradle-init` flag (modeled on the existing
  `--no-auto-approve`).
- Idempotency: if the file already exists and matches in content — no-op; if
  it differs — do not overwrite without `--force` (the user may have edited
  it), and report this. The same principle already adopted in
  `enable_lsp_at_path`.
- Removal: `deep-worker lsp disable` deletes the file, but **only if** its
  content matches the reference (otherwise — warn and leave it).
- Permissions: file `0o644`, directory `0o755`. This is not a secret but
  input data for a foreign process (the Gradle daemon) run by the same user;
  `0600/0700` are not needed here, and `0777` is forbidden by policy. The
  difference from `mcp.json` (`0600/0700`) is intentional and explained by the
  fact that the latter may contain `${env:...}` references to secrets, while
  here there are none.
- Atomic write (tmp + `os.replace`), as in `enable.py`.
- Version marker: as the script's first line — a comment
  `// deep-worker:kotlin-android-source-sets v<N>`, so that "ours/theirs"
  comparison and future migrations do not depend on byte-for-byte equality of
  the whole file.

## 8. Kotlin LSP version

The upgrade to **v262.9593.0** is needed regardless of §4 — v262.2310.0 lacks
working `textDocument/implementation` functionality (previously
`find_implementations` returned `SolidLSPException -32803`). After the upgrade
the server reports the full set:

```
serverInfo: {'name': 'IntelliJ Language Server by JetBrains', 'version': 'LS-262.9593.0'}
  referencesProvider: True
  definitionProvider: True
  implementationProvider: True
  documentSymbolProvider: True
  typeDefinitionProvider: True
```

**How it is pinned.** The stock SolidLSP loader only downloads `.zip`, while
linux-x64 starting with v262.4739.0 is distributed as `.tar.gz` — i.e.
`kotlin_lsp_version` **does not work** for new versions. The working path is
`ls_path` pointing at the unpacked `bin/intellij-server`; this is the
documented public contract of `LanguageServerDependencyProviderBaseCommand`,
implemented in the base class, not in the Kotlin-specific one:

```
* ``ls_path``: overrides the path of the language server's core dependency
  (e.g. its executable or a JAR file), from which the base command is formed,
  bypassing Serena's managed installation
```

Verified end-to-end through **stock** Serena (not a hand-written client):

```
INFO ... solidlsp.language_servers.kotlin_language_server:window_log_message
  LSP: window/logMessage: {'type': 3, 'message': 'System Info
    java.home: .../kotlin_server-262.9593.0/jbr
    java.version: = 25.0.2+10-b432.48
    java.vm.vendor: JetBrains s.r.o.'}
INFO ... Kotlin LSP progress [...]: started - Importing project
INFO ... Kotlin LSP progress [...]: ended -
```

The bundled JBR 25 removes the "JDK 25 required on the system" requirement
(the system only has JDK 17).

**Version scope.** Since `project.yml` does not belong to us (§7, constr. B),
pinning the version "in the repository" is only possible as a **documented
instruction** in `lsp/README.md` (exact `ls_path`, where to get the archive,
why not `kotlin_lsp_version`). I do not take automatic installation of the
server itself into scope: that is a separate task with a ~400 MB download, SHA
verification, and platform selection. Moved to §11 as follow-up F4.

## 9. Scope

**In scope:**
1. The resource script `kotlin-android-source-sets.init.gradle` in the package.
2. The install/uninstall/check module
   (`deep_worker_cli/lsp/gradle_init.py`) with atomic write and permissions
   per §7.1.
3. Wiring into `lsp enable` / `lsp disable` / `lsp status` (+ the
   `--no-gradle-init` flag).
4. Packaging the resource into the wheel (`shared-data`).
5. A section in `lsp/README.md`: what is installed, where, how to
   disable/remove, how to pin `ls_path` for the Kotlin LSP.
6. Unit and integration tests (§10).

**Out of scope (see §11):** auto-installation of the Kotlin LSP binary; fixing
`workspaceSymbol`; the separate `job_id` defect in the sandboxed `execute`;
managing `project.yml`/`serena_config.yml`.

## 10. TDD plan (RED → GREEN + mutation testing)

Unit tests — `deep_worker_cli/tests/unit_tests/lsp/test_gradle_init.py`.

| # | Test | RED without the fix |
|---|---|---|
| T1 | `install_gradle_init` creates the file in the given `init.d`, mode `0o644`, directory `0o755`, content matches the resource | function absent |
| T2 | Idempotency: a second call does not change mtime and returns "not changed"; with differing content and no `force` — does not overwrite and signals | — |
| T3 | The resource script contains the `IdeaGradleLspPlugin` guard **before** any state mutation (static check of the resource text) | — |
| T4 | Every reflective block of the resource is wrapped in `try`/`catch` (static check: no `loadClass`/`getDeclaredMethod` outside `try`) | — |
| T5 | The wrapper marks only tasks with `outcome == null` (otherwise the `assert` from §6.1 is violated) — resource-text check for the `!st.executed` condition | — |
| T6 | `uninstall_gradle_init` removes only "our" file (by version marker), leaves a foreign one and reports | — |
| T7 | Atomicity: on a write error no tmp file remains | — |
| T8 | `lsp enable --no-gradle-init` does not create the file; `lsp enable` does (command-level test) | — |

**Mutation testing** (mandatory, following the precedent of previous fixes):
for T1, T3, T6 introduce a targeted mutation (break the permission mode;
remove the guard from the resource; remove the marker check on removal) and
make sure the corresponding test goes red. Otherwise the test does not prove
the property.

**Integration test** (`tests/integration_tests/lsp/`, no mocks):
`deep-worker lsp enable --scope ...` with a temporary `HOME` → the file
appears; `disable` → it is removed. I do not run a real Gradle in CI: it
requires JDK, network, and about a minute — instead I record the live check
from §12, performed manually. The integration test budget is strict, per the
rule "integ tests must not hang".

## 11. Follow-ups (separate tickets, NOT in this scope)

**F1. `workspaceSymbol` returns results without `uri`/`range`.**
After the fix `workspace/symbol("CrashlyticsLogger")` returns 20 elements, but
the location fields are empty:

```
### workspaceSymbol('CrashlyticsLogger')
   20 result(s):
    - None line None
    - None line None
    ...
```

Looks like a mismatch between the response shape (`WorkspaceSymbol` with
`location: {uri}` instead of a full `Location`) and the SolidLSP parser's
expectations. It is unrelated to this fix: the count is non-zero, so the index
is populated. A separate ticket is needed.

**F2. The `empty set of source sets` line remains after the fix — analyzed.**
It is important here to correct my own earlier verbal explanation: the message
does **not** refer to the MPP model. It is emitted in
`IdeaProjectMapper.splitModulePerSourceSet` from exactly the same map that §5
fixes:

```
183: invokevirtual   ProjectMetadata.getSourceSets()      → Map
187: invokeinterface IdeaModule.getName()
192: invokeinterface Map.get(Object)
216: invokeinterface Collection.isEmpty()
233: getfield        LOG
242: invokedynamic   makeConcatWithConstants  // " has an empty set of source sets"
247: invokevirtual   Logger.info(String)
```

The key is a **quantitative** before/after comparison:

```
before fix : SampleApp has an empty set of source sets
            SampleApp.app has an empty set of source sets      ← 2 lines
after fix  : SampleApp has an empty set of source sets          ← 1 line
```

The line about the `app` module **disappeared** — exactly when the source
roots appeared. The remaining one refers to the root project `SampleApp`,
which is an empty Gradle aggregator (`rootProject.name = "SampleApp";
include(":app")`, `build.gradle.kts` contains only
`plugins { ... apply false }`) — it has no sources and should not. The level
is `INFO`; behavior is unaffected. Confirmation — the `app` module is
correctly split by source sets in the model:
`SampleApp.app.debug`, `SampleApp.app.debugUnitTest`,
`SampleApp.app.debugAndroidTest`. No action needed; recorded to avoid
re-investigating.

**F3. The fourth `job_id` locus in the sandboxed `execute`.** Reproduced in
this same session, unrelated to Kotlin. Documented as a separate design,
`docs/DESIGN_FIX_SANDBOX_EXECUTE_JOB_ID.md`.

**F4. Auto-installation of the Kotlin LSP binary (tar.gz/vsix).** Currently
the server installation is manual, in the README. Automation requires a loader
with SHA verification and platform selection — a separate task (§8).

## 12. Live check (performed before writing the design)

Project SampleApp, an "interface + implementation in different files" pair:
`framework/billing/CrashlyticsLogger.kt` ↔
`framework/billing/FirebaseCrashlyticsLogger.kt`.

| Check | Before | After |
|---|---|---|
| source roots in the workspace model | 3 (no sources) | **21**, including `app/src/main/java`, KSP, BuildConfig |
| `references(CrashlyticsLogger)` | `[]` | **38** hits in 6+ files |
| `implementation(CrashlyticsLogger)` | `None` | `FirebaseCrashlyticsLogger.kt:21` |
| `definition` of supertype cross-file | `[]` | `CrashlyticsLogger.kt:16` |
| `references(logPurchaseError)` | `[]` | **7** hits |
| `workspaceSymbol` | `[]` | 20 (with the F1 caveat) |
| `Querying the mapped value ...` in the log | present | **absent** |
| `Cannot call Task.finalizedBy ...` | — | **absent** (window narrowed, §6.1) |

A fragment of the actual response:

```
### references(CrashlyticsLogger @ CrashlyticsLogger.kt:17)
   38 result(s):
    - app/src/main/java/.../billing/FirebaseCrashlyticsLogger.kt line 21
    - app/src/main/java/.../viewmodels/SubscriptionViewModel.kt line 14
    - app/src/test/java/.../SubscriptionErrorMessagesTest.kt line 5
    - app/build/generated/ksp/debug/java/.../SubscriptionViewModel_Factory.java line 5
    ...
```

## 13. Decisions on open questions (closed by the review lead)

1. **Install default — YES, install at `lsp enable` by default**, opt-out via
   `--no-gradle-init`. Reviewer's rationale: it closes the original
   "works for anyone who clones" claim; the script is still a no-op outside
   `IdeaGradleLspPlugin`.
2. **Conditionality — NO, install unconditionally.** Android-project
   detection is fragile on monorepos, and the script self-neutralizes anyway.
3. **Upstream issue in `Kotlin/kotlin-lsp` — YES, but as a separate activity**
   after shipping, with a reference to this document. It is not a gate for
   code.

**Additional reviewer requirements (not blockers, accepted for
implementation):**

* §6.1 was deemed sufficient as is. Optionally (not required) — catch
  `AssertionError`/`IllegalStateException` around `setOutcome`. **Accepted for
  implementation**: it costs nothing — the same `try/catch` that is already
  required by §6.2.
* The README must explicitly state: the script goes into the **global**
  `~/.gradle/init.d`, and how to remove it.
