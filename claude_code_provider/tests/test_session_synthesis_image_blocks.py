# pyright: reportPrivateUsage=false, reportUnusedFunction=false
"""TDD for image content blocks in synthesized Claude transcripts.

See docs/DESIGN_FIX_IMAGE_BLOCKS_IN_SYNTHESIZED_TRANSCRIPT.md.

THE DEFECT (production, observed 2026-08-04)
─────────────────────────────────────────────────────────
``read_image`` legitimately returns LangChain (OpenAI-style) multimodal blocks::

    [{"type": "text", ...},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

``session_synthesis._tool_result_content`` wrote that list into the transcript
VERBATIM, so the OpenAI-style block landed inside ``tool_result.content`` and
every ``--resume`` replayed it to the Messages API::

    API Error: 400 messages.18.content.0.tool_result.content.1: Input tag
    'image_url' found using 'type' does not match any of the expected tags:
    'document', 'image', 'search_result', 'text', 'tool_reference'

The session was then poisoned permanently — even ``2+2`` failed.

GROUND TRUTH for the target shape was captured from a REAL CLI-written
transcript (``<uuid>.jsonl`` line 9, CLI 2.1.220, produced by Claude's own
native ``Read`` tool)::

    {"type": "image",
     "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}}

⚠️ TWO DIFFERENT IMAGE ENCODINGS EXIST — DO NOT CONFLATE THEM:
  * live tool result → MCP shape ``{"type":"image","data":…,"mimeType":…}``
    (built by ``chat_model._build_mcp_content``; that path is CORRECT and is
    NOT what this module fixes);
  * synthesized transcript → Anthropic shape ``{"type":"image","source":{…}}``.
Reusing ``_build_mcp_content`` here would still 400 (``mimeType`` is not a
valid Messages API key).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import BaseMessage, ToolMessage

from claude_code_provider.session_synthesis import (
    session_project_dir,
    synthesize_session_file,
)


# ===========================================================================
# helpers
# ===========================================================================

def _read_conversation(path: Path) -> list[dict[str, Any]]:
    entries = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    return [e for e in entries if e.get("type") != "queue-operation"]


def _png_bytes(width: int = 4, height: int = 3) -> bytes:
    """A REAL png, so magic-number validation is exercised honestly.

    Using a fake ``b"AAA"`` payload would let a broken magic check pass.
    """
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _data_url(raw: bytes, media_type: str = "image/png") -> str:
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{media_type};base64,{b64}"


def _image_url_block(raw: bytes, media_type: str = "image/png") -> dict[str, Any]:
    """Exactly the shape ``deep_worker_cli.tools.read_image`` returns."""
    return {
        "type": "image_url",
        "image_url": {"url": _data_url(raw, media_type)},
    }


def _tool_result_blocks(path: Path) -> list[Any]:
    """The ``tool_result`` content of the single ToolMessage entry."""
    entries = _read_conversation(path)
    tool_entries = [
        e
        for e in entries
        if isinstance(e.get("message"), dict)
        and isinstance(e["message"].get("content"), list)
        and any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in e["message"]["content"]
        )
    ]
    assert tool_entries, "expected a tool_result entry in the transcript"
    block = next(
        b
        for b in tool_entries[0]["message"]["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )
    return block["content"]


def _synth(msgs: list[BaseMessage], tmp_path: Path) -> Path:
    sid = synthesize_session_file(msgs, cwd="/x")
    assert sid is not None
    return session_project_dir("/x") / f"{sid}.jsonl"


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # pyright: ignore[reportUnknownLambdaType]


# ===========================================================================
# RED — these fail before the fix
# ===========================================================================

class TestNoImageUrlLeaksIntoTranscript:
    """The regression guard for the production 400."""

    def test_synthesized_tool_result_has_no_image_url(self, tmp_path: Path) -> None:
        """No ``image_url`` may appear ANYWHERE in the written transcript.

        This is the direct guard for the verbatim production failure.
        """
        msgs: list[BaseMessage] = [
            ToolMessage(
                content=[
                    {"type": "text", "text": "Image: x.png (4x3), image/png, 70 bytes"},
                    _image_url_block(_png_bytes()),
                ],
                tool_call_id="toolu_01",
            )
        ]
        path = _synth(msgs, tmp_path)

        raw = path.read_text()
        assert "image_url" not in raw, (
            "OpenAI-style image_url leaked into the synthesized transcript — "
            "this is the exact payload that produced the production 400"
        )

    def test_synthesized_tool_result_uses_anthropic_image_shape(
        self, tmp_path: Path
    ) -> None:
        """The block must match the shape captured from a REAL CLI transcript."""
        png = _png_bytes()
        msgs: list[BaseMessage] = [
            ToolMessage(
                content=[
                    {"type": "text", "text": "Image: x.png"},
                    _image_url_block(png),
                ],
                tool_call_id="toolu_01",
            )
        ]
        blocks = _tool_result_blocks(_synth(msgs, tmp_path))

        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "Image: x.png"}
        assert blocks[1] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png).decode("ascii"),
            },
        }

    def test_string_content_unchanged(self, tmp_path: Path) -> None:
        """The dominant case (11,715 of 25,455 real tool_results) must not move."""
        msgs: list[BaseMessage] = [
            ToolMessage(content="plain text result", tool_call_id="toolu_01")
        ]
        entries = _read_conversation(_synth(msgs, tmp_path))
        block = entries[0]["message"]["content"][0]
        assert block["content"] == "plain text result"

    def test_mixed_list_preserves_order_and_count(self, tmp_path: Path) -> None:
        msgs: list[BaseMessage] = [
            ToolMessage(
                content=[
                    {"type": "text", "text": "First"},
                    _image_url_block(_png_bytes()),
                    {"type": "text", "text": "Second"},
                ],
                tool_call_id="toolu_01",
            )
        ]
        blocks = _tool_result_blocks(_synth(msgs, tmp_path))
        assert [b["type"] for b in blocks] == ["text", "image", "text"]
        assert blocks[0]["text"] == "First"
        assert blocks[2]["text"] == "Second"

    def test_real_read_image_output_round_trips(self, tmp_path: Path) -> None:
        """Feed ``read_image``'s LITERAL return value through synthesis.

        Guards against the two shapes drifting apart: if ``read_image`` changes
        its output, this test — not production — is what breaks.
        """
        import sys

        sys.path.insert(0, str(Path("deep_worker/deep_worker_cli").resolve()))
        from deep_worker_cli.tools import read_image  # noqa: PLC0415

        img = tmp_path / "shot.png"
        img.write_bytes(_png_bytes(8, 8))

        result = read_image(str(img))
        assert isinstance(result, list), f"read_image failed: {result!r}"

        # ``read_image`` returns list[dict[str, Any]]; ToolMessage's declared
        # content type is narrower (list[str | dict[Unknown, Unknown]]) and
        # list is invariant, so cast for the type checker only — the runtime
        # value is exactly what production passes.
        msgs: list[BaseMessage] = [
            ToolMessage(
                content=cast("list[str | dict[Any, Any]]", result),
                tool_call_id="toolu_01",
            )
        ]
        path = _synth(msgs, tmp_path)

        assert "image_url" not in path.read_text()
        blocks = _tool_result_blocks(path)
        images = [b for b in blocks if isinstance(b, dict) and b.get("type") == "image"]
        assert len(images) == 1
        assert images[0]["source"]["type"] == "base64"
        assert images[0]["source"]["media_type"] == "image/png"


# ===========================================================================
# Fail-closed validation (design §5.1)
#
# These test the converter directly: they assert that a doubtful payload is
# NEVER emitted as an image block. A converter that merely reshapes whatever
# arrives would relocate the poisoning instead of removing it.
# ===========================================================================

from claude_code_provider.session_synthesis import (  # noqa: E402
    _MAX_BASE64_IMAGE_BYTES,
    _tool_result_content,
)


def _convert_one(block: dict[str, Any]) -> dict[str, Any]:
    out = _tool_result_content([block])
    assert isinstance(out, list) and len(out) == 1
    return out[0]


def _assert_rejected(result: dict[str, Any], payload: str = "") -> None:
    """Rejection must be a text block that leaks NO image bytes."""
    assert result["type"] == "text", f"expected rejection, got {result['type']!r}"
    assert result["text"].startswith("[Image omitted from restored history:")
    if payload:
        assert payload not in result["text"]
        # no long slice of the payload either
        for i in range(0, max(1, len(payload) - 32), 32):
            assert payload[i : i + 32] not in result["text"]
    assert len(result["text"]) < 200, "diagnostic must stay small"


class TestMediaTypeWhitelist:
    @pytest.mark.parametrize(
        "media_type",
        ["image/svg+xml", "image/bmp", "application/octet-stream", "text/html", ""],
    )
    def test_unsupported_media_type_rejected(self, media_type: str) -> None:
        b64 = base64.b64encode(_png_bytes()).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        }
        _assert_rejected(_convert_one(block), b64)

    def test_media_type_is_case_and_parameter_insensitive(self) -> None:
        png = _png_bytes()
        b64 = base64.b64encode(png).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:IMAGE/PNG;charset=utf-8;base64,{b64}"},
        }
        assert _convert_one(block)["type"] == "image"

    @pytest.mark.parametrize(
        ("media_type", "fmt"),
        [("image/png", "PNG"), ("image/jpeg", "JPEG"), ("image/gif", "GIF"),
         ("image/webp", "WEBP")],
    )
    def test_all_supported_types_accepted(self, media_type: str, fmt: str) -> None:
        """Real encoded bytes per format, so magic validation is honest."""
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4), (0, 128, 255)).save(buf, format=fmt)
        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        }
        result = _convert_one(block)
        assert result["type"] == "image", result
        assert result["source"]["media_type"] == media_type


class TestMagicNumberSpoofGuard:
    def test_declared_png_but_actually_jpeg_rejected(self) -> None:
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4)).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        _assert_rejected(_convert_one(block), b64)

    @pytest.mark.parametrize(
        ("form_type", "label"),
        [(b"WAVE", "RIFF/WAVE audio"), (b"AVI ", "RIFF/AVI video")],
    )
    def test_riff_container_that_is_not_webp_rejected(
        self, form_type: bytes, label: str
    ) -> None:
        """WebP is RIFF + 'WEBP' at [8:12] — a bare RIFF prefix is NOT enough.

        A bare ``startswith(b"RIFF")`` check would accept WAVE/AVI as an image.
        The rejection must come from the magic guard, i.e. BEFORE Pillow.
        """
        payload = b"RIFF" + (100).to_bytes(4, "little") + form_type + b"\x00" * 88
        b64 = base64.b64encode(payload).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/webp;base64,{b64}"},
        }
        result = _convert_one(block)
        _assert_rejected(result, b64)
        assert "media type mismatch" in result["text"], (
            f"{label} must be rejected by the magic guard, not by Pillow"
        )

    def test_riff_webp_magic_rejected_before_pillow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The RIFF/WAVE spoof must never reach Pillow."""
        import claude_code_provider.session_synthesis as mod

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError("Image.open must NOT be reached on magic mismatch")

        monkeypatch.setattr(mod.Image, "open", _boom)
        payload = b"RIFF" + (100).to_bytes(4, "little") + b"WAVE" + b"\x00" * 88
        b64 = base64.b64encode(payload).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/webp;base64,{b64}"},
        }
        assert "media type mismatch" in _convert_one(block)["text"]

    def test_declared_png_but_arbitrary_bytes_rejected(self) -> None:
        b64 = base64.b64encode(b"not an image at all").decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        _assert_rejected(_convert_one(block), b64)


class TestBase64Validation:
    def test_invalid_alphabet_rejected(self) -> None:
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,!!!!not-base64!!!!"},
        }
        _assert_rejected(_convert_one(block))

    def test_bad_padding_rejected(self) -> None:
        b64 = base64.b64encode(_png_bytes()).decode("ascii").rstrip("=") + "A"
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        _assert_rejected(_convert_one(block))

    def test_empty_data_rejected(self) -> None:
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,"},
        }
        _assert_rejected(_convert_one(block))

    def test_non_ascii_data_rejected_without_unicode_error(self) -> None:
        """A smart quote must degrade, not raise UnicodeEncodeError."""
        block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBOR“w0KGgo"},
        }
        _assert_rejected(_convert_one(block))


class TestEncodedSizeLimit:
    """The limit is on the BASE64-ENCODED payload (Anthropic Vision docs)."""

    def test_encoded_exactly_at_cap_is_not_rejected_for_size(self) -> None:
        """At the cap, size must not be the rejection reason."""
        data = "A" * _MAX_BASE64_IMAGE_BYTES
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }
        result = _convert_one(block)
        # It fails for other reasons (not a real png) but NOT for size.
        assert "too large" not in result["text"]

    def test_encoded_over_cap_rejected(self) -> None:
        data = "A" * (_MAX_BASE64_IMAGE_BYTES + 1)
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }
        result = _convert_one(block)
        assert "too large" in result["text"]

    def test_decoded_under_10mb_but_encoded_over_cap_is_rejected(self) -> None:
        """THE decisive case for the encoded-vs-decoded contract.

        A payload whose DECODED size is comfortably under 10 MB but whose
        ENCODED size exceeds the cap. Under a decoded-based limit this would be
        accepted and could then be rejected by the API — re-poisoning the
        session. It must be rejected here.
        """
        encoded_len = _MAX_BASE64_IMAGE_BYTES + 4
        decoded_len = encoded_len // 4 * 3
        assert decoded_len < 10_000_000, "decoded must be under 10 MB for this test"
        data = "A" * encoded_len
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }
        assert "too large" in _convert_one(block)["text"]

    def test_oversize_rejected_without_decoding_or_pillow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves the budget is ENFORCED, not merely claimed.

        An oversized payload must be rejected before ``b64decode`` or Pillow is
        ever reached.
        """
        import claude_code_provider.session_synthesis as mod

        def _boom_decode(*_a: Any, **_kw: Any) -> bytes:
            raise AssertionError("b64decode must NOT be called for oversize input")

        def _boom_open(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError("Image.open must NOT be called for oversize input")

        monkeypatch.setattr(mod.base64, "b64decode", _boom_decode)
        monkeypatch.setattr(mod.Image, "open", _boom_open)

        data = "A" * (_MAX_BASE64_IMAGE_BYTES + 1)
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }
        assert "too large" in _convert_one(block)["text"]


class TestDimensionAndBombGuard:
    def test_oversized_dimensions_rejected(self) -> None:
        from PIL import Image as PILImage

        buf = io.BytesIO()
        # 9000 px exceeds the 8000 px limit on one axis.
        PILImage.new("L", (9000, 4)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        assert "dimensions too large" in _convert_one(block)["text"]

    def test_decompression_bomb_warning_band_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pillow only WARNS between MAX_IMAGE_PIXELS and 2x that.

        Verified in Pillow 12.0.0 ``Image._decompression_bomb_check``. An image
        in that band must still be rejected by us — relying on the exception
        alone would let it through.
        """
        import claude_code_provider.session_synthesis as mod

        # Shrink the threshold so the band is cheap to construct: a 100x100
        # image then sits between MAX_IMAGE_PIXELS and 2x.
        monkeypatch.setattr(mod.Image, "MAX_IMAGE_PIXELS", 8000)

        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("L", (100, 100)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
        _assert_rejected(_convert_one(block), b64)


class TestIdempotenceDoesNotPreservePoison:
    def test_valid_already_anthropic_block_passes(self) -> None:
        png = _png_bytes()
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png).decode("ascii"),
            },
        }
        assert _convert_one(block) == block

    @pytest.mark.parametrize(
        "source",
        [
            {"type": "url", "url": "https://example.com/x.png"},
            {"type": "base64", "media_type": "image/svg+xml", "data": "PHN2Zz48L3N2Zz4="},
            {"type": "base64", "media_type": "image/png"},
            {"type": "base64", "data": "AAAA"},
        ],
    )
    def test_malformed_already_anthropic_block_rejected(
        self, source: dict[str, Any]
    ) -> None:
        """Unconditional pass-through would CONSERVE poison — it must not."""
        _assert_rejected(_convert_one({"type": "image", "source": source}))

    def test_mcp_shape_normalized_to_source(self) -> None:
        png = _png_bytes()
        block = {
            "type": "image",
            "data": base64.b64encode(png).decode("ascii"),
            "mimeType": "image/png",
        }
        result = _convert_one(block)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert "mimeType" not in result

    def test_malformed_mcp_shape_rejected(self) -> None:
        _assert_rejected(_convert_one({"type": "image", "data": "AAAA"}))


class TestSchemaPolicy:
    @pytest.mark.parametrize(
        "block",
        [
            {"type": "text", "text": "hello"},
            {"type": "document", "source": {"type": "text", "data": "x"}},
            {"type": "search_result", "title": "t"},
            {"type": "tool_reference", "name": "n"},
        ],
    )
    def test_supported_non_image_types_pass_through_unchanged(
        self, block: dict[str, Any]
    ) -> None:
        """API-supported types must NOT be flattened to text."""
        assert _convert_one(block) == block

    def test_unknown_dict_block_becomes_text(self) -> None:
        result = _convert_one({"type": "totally_unknown", "x": 1})
        assert result["type"] == "text"

    def test_non_dict_block_becomes_text(self) -> None:
        out = _tool_result_content(["bare string", 42])
        assert isinstance(out, list)
        assert all(b["type"] == "text" for b in out)

    def test_non_data_url_degrades_to_text(self) -> None:
        block = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.png"},
        }
        result = _convert_one(block)
        assert result == {"type": "text", "text": "[Image: https://example.com/cat.png]"}

    def test_image_url_as_bare_string_supported(self) -> None:
        """LangChain also allows image_url to be a plain string."""
        b64 = base64.b64encode(_png_bytes()).decode("ascii")
        block = {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
        assert _convert_one(block)["type"] == "image"

    def test_empty_list_degrades_to_empty_string(self) -> None:
        """An empty content list is invalid for the API."""
        assert _tool_result_content([]) == ""

    def test_converter_never_raises(self) -> None:
        """Restore is an optimization; it must never break a turn."""
        for weird in [
            {"type": "image_url"},
            {"type": "image_url", "image_url": None},
            {"type": "image_url", "image_url": {"url": None}},
            {"type": "image", "source": None},
            {"type": "image"},
            None,
        ]:
            out = _tool_result_content([weird])
            assert isinstance(out, list) and len(out) == 1
