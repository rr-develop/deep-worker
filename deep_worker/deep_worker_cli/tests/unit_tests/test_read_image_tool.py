"""Tests for the read_image tool."""

import base64
import json
import tempfile
from pathlib import Path

import pytest

from PIL import Image


# ===========================================================================
# read_image tool tests
# ===========================================================================

class TestReadImage:
    """Tests for the read_image tool function."""

    @pytest.fixture
    def png_file(self, tmp_path: Path) -> Path:
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        p = tmp_path / "test.png"
        img.save(str(p))
        return p

    @pytest.fixture
    def jpg_file(self, tmp_path: Path) -> Path:
        img = Image.new("RGB", (10, 10), (0, 255, 0))
        p = tmp_path / "test.jpg"
        img.save(str(p))
        return p

    def test_read_png_returns_image_content(self, png_file: Path):
        from deep_worker_cli.tools import read_image

        result = read_image(str(png_file))
        assert isinstance(result, list)
        assert len(result) == 2

        # First block: text metadata
        assert result[0]["type"] == "text"
        assert "test.png" in result[0]["text"]

        # Second block: image data
        assert result[1]["type"] == "image_url"
        url = result[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify base64 is valid
        b64_data = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    def test_read_jpeg_correct_mime(self, jpg_file: Path):
        from deep_worker_cli.tools import read_image

        result = read_image(str(jpg_file))
        url = result[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_file_not_found(self):
        from deep_worker_cli.tools import read_image

        result = read_image("/nonexistent/image.png")
        assert isinstance(result, str)
        assert "not found" in result.lower() or "error" in result.lower()

    def test_unsupported_format(self, tmp_path: Path):
        from deep_worker_cli.tools import read_image

        bmp = tmp_path / "test.bmp"
        bmp.write_bytes(b"BM" + b"\x00" * 100)
        result = read_image(str(bmp))
        assert isinstance(result, str)
        assert "unsupported" in result.lower()

    def test_file_too_large(self, tmp_path: Path):
        from deep_worker_cli.tools import read_image

        big = tmp_path / "big.png"
        # Write 21MB file (over 20MB limit)
        big.write_bytes(b"\x89PNG" + b"\x00" * (21 * 1024 * 1024))
        result = read_image(str(big))
        assert isinstance(result, str)
        assert "too large" in result.lower() or "maximum" in result.lower()

    def test_webp_supported(self, tmp_path: Path):
        from deep_worker_cli.tools import read_image

        img = Image.new("RGB", (4, 4), (0, 0, 255))
        p = tmp_path / "test.webp"
        img.save(str(p))
        result = read_image(str(p))
        assert isinstance(result, list)
        url = result[1]["image_url"]["url"]
        assert url.startswith("data:image/webp;base64,")


# ===========================================================================
# resolve_call multimodal handling tests
# ===========================================================================

# ===========================================================================
# Guard: read_image registered in tools list
# ===========================================================================

class TestReadImageRegistered:
    """Guard test: read_image must be in the agent's tool list."""

    def test_read_image_in_agent_process(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "deep_worker_cli" / "process" / "agent_process.py"
        ).read_text()
        assert "read_image" in source, (
            "read_image must be imported and added to web_tools in agent_process.py"
        )
