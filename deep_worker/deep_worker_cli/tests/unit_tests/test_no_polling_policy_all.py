
"""
Test that NO polling loops exist in deep_worker_cli source.

Scans ONLY the project's own source tree (deep_worker_cli/) for forbidden
polling patterns: while loops containing time.sleep() or
await asyncio.sleep(). Such a loop wakes N times a second to re-ask a
question the kernel could answer by waking us once, and it is the loop
that is forbidden -- not sleep itself. A bare sleep outside a retry loop
is not what this scanner is looking for and does not fail it.

The scan excludes __pycache__ but does NOT need to exclude .venv -- virtual
environments are already forbidden in this repo by the pre-commit hook.
"""
import ast
from pathlib import Path


def find_polling_loops_in_file(filepath: Path) -> list[tuple[str, int, str]]:
    """Return list of (function_name, line_number, message) for polling loops."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    polling_loops = []

    class PollingVisitor(ast.NodeVisitor):
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._check_function(node)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._check_function(node)
            self.generic_visit(node)

        def _check_function(self, node: ast.AsyncFunctionDef | ast.FunctionDef):
            for child in ast.walk(node):
                if not isinstance(child, ast.While):
                    continue
                for subchild in ast.walk(child):
                    # Pattern 1: while + await asyncio.sleep(...)
                    if isinstance(subchild, ast.Await):
                        if (isinstance(subchild.value, ast.Call) and
                            isinstance(subchild.value.func, ast.Attribute) and
                            subchild.value.func.attr == 'sleep' and
                            isinstance(subchild.value.func.value, ast.Name) and
                            subchild.value.func.value.id == 'asyncio'):
                            loop_code = ast.get_source_segment(content, child)
                            polling_loops.append((
                                node.name, child.lineno,
                                f"while loop with await asyncio.sleep at line {child.lineno}:\n{loop_code}"
                            ))
                            break
                    # Pattern 2: while + time.sleep(...)
                    if (isinstance(subchild, ast.Expr) and
                        isinstance(subchild.value, ast.Call) and
                        isinstance(subchild.value.func, ast.Attribute) and
                        subchild.value.func.attr == 'sleep' and
                        isinstance(subchild.value.func.value, ast.Name) and
                        subchild.value.func.value.id == 'time'):
                        loop_code = ast.get_source_segment(content, child)
                        polling_loops.append((
                            node.name, child.lineno,
                            f"while loop with time.sleep at line {child.lineno}:\n{loop_code}"
                        ))
                        break

    PollingVisitor().visit(tree)
    return polling_loops


def test_no_polling_anywhere():
    """Test that NO deep_worker_cli source files contain polling loops."""
    project_root = Path(__file__).parent.parent.parent
    src = project_root / "deep_worker_cli"
    assert src.is_dir(), f"Source directory not found: {src}"

    all_polling_loops = []
    for filepath in sorted(src.rglob("*.py")):
        if "__pycache__" in str(filepath):
            continue
        for func, line, msg in find_polling_loops_in_file(filepath):
            rel = filepath.relative_to(project_root)
            all_polling_loops.append(f"{rel}:{func}:{line}\n{msg}")

    assert all_polling_loops == [], (
        "Found polling loops in the project:\n" +
        "\n".join(f"  {loop}" for loop in all_polling_loops)
    )


if __name__ == "__main__":
    test_no_polling_anywhere()
    print("No polling loops found in the project!")
