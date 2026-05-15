"""
Test that deep_worker_cli has NO polling loops.

Polling loops are patterns like:
    while <condition>: await asyncio.sleep(x)

They waste CPU by waking up even when there's no work to do.
All waiting should be event-driven (asyncio.Event, asyncio.Condition, etc.).
"""

import ast
import asyncio
import inspect
import os
from pathlib import Path
from typing import Any


def contains_polling_loop(source: str) -> bool:
    """Return True if the function body contains a loop that awaits asyncio.sleep().
    
    Patterns we detect:
        while <cond>: await asyncio.sleep(<x>)
        for _ in range(N): await asyncio.sleep(<x>)
        for _ in <iterable>: await asyncio.sleep(<x>)
    
    A bare ``await asyncio.sleep(x)`` outside a loop is fine (tests, pacing a
    send). A ``for _ in range(N): await asyncio.sleep(x)`` is also polling.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    
    class PollingDetector(ast.NodeVisitor):
        def __init__(self):
            self.found_polling = False
        
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            # Don't descend into nested async functions
            pass
        
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Don't descend into nested sync functions
            pass
        
        def visit_While(self, node: ast.While) -> None:
            # Check if while body contains await asyncio.sleep(...)
            self._check_loop_body(node.body)
            self.generic_visit(node)
        
        def visit_For(self, node: ast.For) -> None:
            # Check if for body contains await asyncio.sleep(...)
            self._check_loop_body(node.body)
            self.generic_visit(node)
        
        def _check_loop_body(self, body: list[ast.stmt]) -> None:
            """Check if loop body contains await asyncio.sleep(...)."""
            for stmt in body:
                if self._is_asyncio_sleep_await(stmt):
                    self.found_polling = True
                    return
        
        def _is_asyncio_sleep_await(self, node: ast.stmt) -> bool:
            """Check if statement is 'await asyncio.sleep(...)'."""
            if not isinstance(node, ast.Expr):
                return False
            
            expr = node.value
            if not isinstance(expr, ast.Await):
                return False
            
            await_expr = expr.value
            if not isinstance(await_expr, ast.Call):
                return False
            
            # Check if it's asyncio.sleep(...)
            func = await_expr.func
            if isinstance(func, ast.Attribute):
                if func.attr == "sleep":
                    # Check if it's asyncio.sleep or module.sleep
                    if isinstance(func.value, ast.Name):
                        if func.value.id == "asyncio":
                            return True
                    # Also check for imported asyncio (e.g., import asyncio as aio)
                    elif isinstance(func.value, ast.Attribute):
                        # Handle cases like aio.sleep where asyncio imported as aio
                        return True
            
            return False
    
    detector = PollingDetector()
    detector.visit(tree)
    return detector.found_polling


def find_polling_loops_in_file(filepath: Path) -> list[tuple[str, int, str]]:
    """Find all polling loops in a Python file.
    
    Returns list of (function_name, line_number, source_snippet) tuples.
    """
    results = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return results
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return results
    
    class PollingFinder(ast.NodeVisitor):
        def __init__(self, source: str):
            self.source = source
            self.results: list[tuple[str, int, str]] = []
            self.current_function = None
        
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            old_function = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old_function
        
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            old_function = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old_function
        
        def visit_While(self, node: ast.While) -> None:
            self._check_loop(node, "while")
            self.generic_visit(node)
        
        def visit_For(self, node: ast.For) -> None:
            self._check_loop(node, "for")
            self.generic_visit(node)
        
        def _check_loop(self, node: ast.AST, loop_type: str) -> None:
            """Check if loop contains await asyncio.sleep(...)."""
            # Get loop body source
            lines = self.source.splitlines()
            start_line = node.lineno - 1  # 0-indexed
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 5
            
            # Check each statement in loop body
            for stmt in getattr(node, 'body', []):
                if self._is_asyncio_sleep_await(stmt):
                    # Found polling loop
                    snippet = "\n".join(lines[start_line:end_line + 1])
                    self.results.append((
                        self.current_function or "<module>",
                        node.lineno,
                        f"{loop_type} loop with await asyncio.sleep at line {stmt.lineno}:\n{snippet}"
                    ))
                    break
        
        def _is_asyncio_sleep_await(self, node: ast.stmt) -> bool:
            """Check if statement is 'await asyncio.sleep(...)'."""
            if not isinstance(node, ast.Expr):
                return False
            
            expr = node.value
            if not isinstance(expr, ast.Await):
                return False
            
            await_expr = expr.value
            if not isinstance(await_expr, ast.Call):
                return False
            
            # Check if it's asyncio.sleep(...)
            func = await_expr.func
            if isinstance(func, ast.Attribute):
                if func.attr == "sleep":
                    return True
            
            return False
    
    finder = PollingFinder(source)
    finder.visit(tree)
    return finder.results


def test_no_polling_in_agent_process():
    """agent_process.py MUST NOT busy-wait via asyncio.sleep."""
    filepath = Path(__file__).parent.parent.parent / "deep_worker_cli" / "process" / "agent_process.py"
    polling_loops = find_polling_loops_in_file(filepath)
    
    assert polling_loops == [], (
        f"agent_process.py contains polling loops:\n" +
        "\n".join(f"  {func}:{line}\n{msg}" for func, line, msg in polling_loops)
    )


def test_no_polling_in_app():
    """app.py MUST NOT busy-wait via asyncio.sleep."""
    filepath = Path(__file__).parent.parent.parent / "deep_worker_cli" / "app.py"
    polling_loops = find_polling_loops_in_file(filepath)
    
    assert polling_loops == [], (
        f"app.py contains polling loops:\n" +
        "\n".join(f"  {func}:{line}\n{msg}" for func, line, msg in polling_loops)
    )


def test_no_polling_in_agent_stop():
    """agent_stop.py MUST NOT busy-wait via asyncio.sleep."""
    filepath = Path(__file__).parent.parent.parent / "deep_worker_cli" / "process" / "agent_stop.py"
    polling_loops = find_polling_loops_in_file(filepath)
    
    assert polling_loops == [], (
        f"agent_stop.py contains polling loops:\n" +
        "\n".join(f"  {func}:{line}\n{msg}" for func, line, msg in polling_loops)
    )


def test_no_polling_in_deep_worker_cli():
    """Check all deep_worker_cli Python files for polling loops."""
    base_dir = Path(__file__).parent.parent.parent / "deep_worker_cli"
    polling_files = []
    
    for py_file in base_dir.rglob("*.py"):
        # Skip test files
        if "test" in str(py_file) or "__pycache__" in str(py_file):
            continue
        
        polling_loops = find_polling_loops_in_file(py_file)
        if polling_loops:
            polling_files.append((py_file, polling_loops))
    
    if polling_files:
        error_msg = "Found polling loops in:\n"
        for filepath, loops in polling_files:
            error_msg += f"\n{filepath.relative_to(base_dir)}:\n"
            for func, line, msg in loops:
                error_msg += f"  {func}:{line} - {msg.split(chr(10))[0]}\n"
        
        raise AssertionError(error_msg)


if __name__ == "__main__":
    # Run tests
    test_no_polling_in_agent_process()
    print("✓ agent_process.py passed")
    
    test_no_polling_in_app()
    print("✓ app.py passed")
    
    test_no_polling_in_agent_stop()
    print("✓ agent_stop.py passed")
    
    test_no_polling_in_deep_worker_cli()
    print("✓ All deep_worker_cli files passed")