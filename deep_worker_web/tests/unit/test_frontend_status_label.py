from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

MAIN_JS = Path(__file__).resolve().parents[2] / "static" / "js" / "main.js"


def test_frontend_status_display_label() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    script = f"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync({str(MAIN_JS)!r}, 'utf8');
const context = {{
  console: {{ log() {{}}, error() {{}}, warn() {{}} }},
  document: {{ addEventListener() {{}}, getElementById() {{ return null; }} }},
  window: {{}},
  fetch() {{ throw new Error('fetch not expected'); }},
  WebSocket: function() {{}},
  setTimeout,
  clearTimeout,
}};
vm.createContext(context);
vm.runInContext(code, context);
if (context.getStatusDisplayLabel({{status: 'idle'}}) !== 'idle') {{
  throw new Error('plain idle should render idle');
}}
if (context.getStatusDisplayLabel({{status: 'idle', status_metadata: {{substatus: 'waiting_agent'}}}}) !== 'idle(waiting_agent)') {{
  throw new Error('idle substatus should render idle(waiting_agent)');
}}
if (context.getStatusDisplayLabel({{status: 'running', status_metadata: {{substatus: 'waiting_agent'}}}}) !== 'running') {{
  throw new Error('non-idle status must ignore idle substatus display');
}}
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
