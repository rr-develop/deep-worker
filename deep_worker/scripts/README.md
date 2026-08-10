# Development Scripts

This directory contains essential development scripts for the deep_worker project.

## pre-commit-type-check.sh

**CRITICAL**: Mandatory type checking pre-commit hook for `deep_worker_cli`.

### Purpose

Enforces strict type checking using the `ty` type checker before every commit. This ensures that all code committed to the repository is type-safe and maintains high code quality standards.

### Installation

The hook is automatically installed in `.git/hooks/pre-commit` for this repository.

If you need to reinstall it manually:

```bash
# From repository root
cp deep_worker/scripts/pre-commit-type-check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Requirements

- **uv**: Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ```

### Usage

The hook runs automatically before every commit. No manual intervention is needed.

**If type checking fails:**

1. Review the type errors displayed
2. Fix the errors in your code
3. Verify the fix with `deep_worker/scripts/type-check.sh`:
   ```bash
   deep_worker/scripts/type-check.sh                  # both packages
   deep_worker/scripts/type-check.sh deepagents       # just one, to iterate
   ```
4. Commit again

> Use that wrapper rather than calling `uv` yourself. It deletes the virtual
> environment it creates; invoking the package manager directly leaves a
> `.venv` behind, and the next commit is then rejected by the `.venv` check
> instead of by your type error.
> See [`DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md`](DESIGN_FIX_TYPE_CHECK_VENV_LEAK.md).

### Policy

⚠️ **STRICTLY PROHIBITED:**

- Using `git commit --no-verify` to bypass this check
- Deleting or disabling this hook
- Modifying the script to skip type checking
- Adding `type: ignore` comments without proper justification

**Type safety is mandatory.** If the check fails, fix the type errors - do not bypass the check.

### Technical Details

- **Checks**: `deep_worker_cli` package only (not `deepagents`)
- **Tool**: `ty` type checker
- **Scope**: All staged files in `deep_worker_cli/`
- **Exit codes**:
  - `0` - Type check passed
  - `1` - Type check failed (commit blocked)

### Troubleshooting

**Hook doesn't run:**
- Verify hook is installed: `ls -l .git/hooks/pre-commit`
- Check executable permission: `chmod +x .git/hooks/pre-commit`

**`uv` command not found:**
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Add to PATH: `export PATH="$HOME/.local/bin:$PATH"`

**Type errors in dependencies:**
- This hook only checks `deep_worker_cli`, not `deepagents`
- If deepagents has type errors, they won't block commits
