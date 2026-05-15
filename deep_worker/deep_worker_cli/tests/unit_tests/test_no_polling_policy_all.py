"""
Test that NO polling loops exist anywhere in deep_worker_cli.

This test scans all Python files in the project for forbidden polling patterns.
"""
import ast
import sys
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
            # Check for while loops with await asyncio.sleep inside
            for child in ast.walk(node):
                if isinstance(child, ast.While):
                    # Check if while contains await asyncio.sleep
                    for subchild in ast.walk(child):
                        if isinstance(subchild, ast.Await):
                            # Check if it's asyncio.sleep call
                            if (isinstance(subchild.value, ast.Call) and
                                isinstance(subchild.value.func, ast.Attribute) and
                                subchild.value.func.attr == 'sleep'):
                                # Check if it's from asyncio module
                                if (isinstance(subchild.value.func.value, ast.Name) and
                                    subchild.value.func.value.id == 'asyncio'):
                                    # Found polling loop
                                    loop_code = ast.get_source_segment(content, child)
                                    polling_loops.append((
                                        node.name,
                                        child.lineno,
                                        f"while loop with await asyncio.sleep at line {child.lineno}:\n{loop_code}"
                                    ))
                                    break

    PollingVisitor().visit(tree)
    return polling_loops


def test_no_polling_anywhere():
    """Test that NO Python files in deep_worker_cli contain polling loops."""
    project_root = Path(__file__).parent.parent.parent
    python_files = list(project_root.rglob("*.py"))
    
    # Exclude test files themselves
    python_files = [f for f in python_files if "test_" not in f.name or "test_no_polling" in f.name]
    
    all_polling_loops = []
    
    for filepath in python_files:
        polling_loops = find_polling_loops_in_file(filepath)
        if polling_loops:
            rel_path = filepath.relative_to(project_root)
            for func, line, msg in polling_loops:
                all_polling_loops.append(f"{rel_path}:{func}:{line}\n{msg}")
    
    assert all_polling_loops == [], (
        "Found polling loops in the project:\n" +
        "\n".join(f"  {loop}" for loop in all_polling_loops)
    )


if __name__ == "__main__":
    test_no_polling_anywhere()
    print("✅ No polling loops found in the project!")