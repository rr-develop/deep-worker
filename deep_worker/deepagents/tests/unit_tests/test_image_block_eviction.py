# pyright: reportPrivateUsage=false
"""Image blocks must survive large-tool-result eviction.

Implements the test plan of
`DESIGN_FIX_IMAGE_BLOCKS_SURVIVE_TOOL_RESULT_EVICTION.md` (tests #1-#17 and
#21-#25). Numbering in docstrings refers to that document.
"""

import base64

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.middleware.filesystem import (
    _MAX_PRESERVED_IMAGE_BLOCKS,
    _MAX_PRESERVED_IMAGE_ENCODED_BYTES,
    _MAX_PRESERVED_IMAGE_TOTAL_BYTES,
    FilesystemMiddleware,
    FilesystemState,
)

# A short, obviously-recognisable payload marker. Any leak of preserved or
# rejected image bytes into the offload file or the notice text is detected by
# searching for this string.
PAYLOAD_MARKER = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo"


def make_runtime(tool_call_id: str) -> ToolRuntime:
    state = FilesystemState(messages=[], files={})
    return ToolRuntime(
        state=state,
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def base64_body(encoded_len: int) -> str:
    """A valid base64 string of exactly `encoded_len` characters.

    The middleware measures the encoded payload without decoding it, but it does
    require genuine base64, so the helper must produce a legal 4-char-group
    string rather than arbitrary filler.
    """
    if encoded_len % 4 != 0:
        raise ValueError("base64 length must be a multiple of 4")
    return (PAYLOAD_MARKER * (encoded_len // len(PAYLOAD_MARKER) + 1))[:encoded_len]


def data_url(encoded_len: int, *, media_type: str = "image/jpeg") -> str:
    """Build a data URL whose base64 section is exactly `encoded_len` chars."""
    return f"data:{media_type};base64,{base64_body(encoded_len)}"


def image_url_block(encoded_len: int = 200_000) -> dict:
    """A LangChain `image_url` block, the shape `read_image` actually emits."""
    return {"type": "image_url", "image_url": {"url": data_url(encoded_len)}}


def anthropic_image_block(encoded_len: int = 200_000) -> dict:
    """Anthropic transcript shape."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_body(encoded_len)},
    }


def mcp_image_block(encoded_len: int = 200_000) -> dict:
    """MCP shape."""
    return {"type": "image", "data": base64_body(encoded_len), "mimeType": "image/png"}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def content_of(result) -> object:
    """Extract the resulting ToolMessage content from a ToolMessage or Command."""
    if isinstance(result, Command):
        return result.update["messages"][0].content
    return result.content


def files_of(result) -> dict:
    if isinstance(result, Command):
        return result.update.get("files") or {}
    return {}


def image_blocks_in(content) -> list:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") in ("image_url", "image")]


def all_text_of(content) -> str:
    """Flatten every textual scrap of a result so leaks cannot hide."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif not (isinstance(block, dict) and block.get("type") in ("image_url", "image")):
            parts.append(str(block))
    return "\n".join(parts)


def file_text(result, tool_call_id: str) -> str:
    files = files_of(result)
    key = f"/large_tool_results/{tool_call_id}"
    if key not in files:
        return ""
    return "\n".join(files[key]["content"])


class TestImageBlocksSurviveEviction:
    """§5.1 block partitioning."""

    def test_red_read_image_shaped_result_keeps_image_block(self):
        """#1 RED: the actual `read_image` shape must keep its image block."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_red")

        content = [
            text_block("Image: photo.jpg (576x1280), image/jpeg, 96840 bytes"),
            image_url_block(129_000),
        ]
        message = ToolMessage(content=content, tool_call_id="test_red")
        result = middleware._intercept_large_tool_result(message, runtime)

        preserved = image_blocks_in(content_of(result))
        assert len(preserved) == 1, "the image block was destroyed by eviction"
        assert preserved[0]["type"] == "image_url"
        assert preserved[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_large_text_only_list_still_evicted(self):
        """#2 regression pin: image-free list content evicts exactly as before."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_textlist")

        content = [text_block("First block " * 500), text_block("Second block " * 500)]
        message = ToolMessage(content=content, tool_call_id="test_textlist")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert isinstance(result, Command)
        assert "/large_tool_results/test_textlist" in files_of(result)
        assert "Tool result too large" in str(content_of(result))
        assert file_text(result, "test_textlist").startswith("[{")

    def test_large_plain_string_still_evicted(self):
        """#3 regression pin: plain string eviction is untouched."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_str")

        message = ToolMessage(content="x" * 5000, tool_call_id="test_str")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert isinstance(result, Command)
        assert file_text(result, "test_str") == "x" * 5000

    def test_small_content_returned_unchanged(self):
        """#4: below threshold, the original object is returned."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        runtime = make_runtime("test_small")

        content = [text_block("small"), image_url_block(100)]
        message = ToolMessage(content=content, tool_call_id="test_small")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert result is message
        assert result.content == content

    def test_image_plus_huge_text_offloads_text_preserves_image(self):
        """#5 (§5.2 row 5): large evictable text offloads, image rides along."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_mix")

        content = [text_block("LOG " * 5000), image_url_block(50_000)]
        message = ToolMessage(content=content, tool_call_id="test_mix")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert isinstance(result, Command)
        out = content_of(result)
        assert len(image_blocks_in(out)) == 1
        assert "Tool result too large" in all_text_of(out)
        assert "/large_tool_results/test_mix" in files_of(result)

    def test_offloaded_file_does_not_contain_base64_payload(self):
        """#6: the base64 payload must never land in the offload file."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_leak")

        content = [text_block("LOG " * 5000), image_url_block(50_000)]
        message = ToolMessage(content=content, tool_call_id="test_leak")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert PAYLOAD_MARKER not in file_text(result, "test_leak")
        assert PAYLOAD_MARKER not in all_text_of(content_of(result))

    @pytest.mark.parametrize(
        ("label", "block_factory"),
        [("anthropic", anthropic_image_block), ("mcp", mcp_image_block)],
    )
    def test_alternate_image_shapes_preserved(self, label, block_factory):
        """#7: Anthropic and MCP image shapes are preserved too."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        tool_call_id = f"test_shape_{label}"
        runtime = make_runtime(tool_call_id)

        content = [text_block("LOG " * 5000), block_factory(50_000)]
        message = ToolMessage(content=content, tool_call_id=tool_call_id)
        result = middleware._intercept_large_tool_result(message, runtime)

        assert len(image_blocks_in(content_of(result))) == 1
        assert PAYLOAD_MARKER not in file_text(result, tool_call_id)

    def test_non_data_url_left_in_evictable_partition(self):
        """#11: remote URLs are not preserved and not fetched."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_remote")

        content = [
            text_block("LOG " * 5000),
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]
        message = ToolMessage(content=content, tool_call_id="test_remote")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert isinstance(result, Command)
        # Not preserved as an image block: there is no inline payload, and the
        # middleware must not fetch the URL.
        assert image_blocks_in(content_of(result)) == []
        # The block travels into the offloaded text EXACTLY as it arrived:
        # deepagents must not decide a URL is unusable for every provider.
        assert str(content[1]) in file_text(result, "test_remote")

    def test_remote_url_in_anthropic_image_shape_is_not_lost(self):
        """A remote URL carried by the `image` shape must survive into the file."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_remote2")

        content = [
            text_block("LOG " * 5000),
            {"type": "image", "url": "https://example.com/b.png"},
        ]
        message = ToolMessage(content=content, tool_call_id="test_remote2")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert image_blocks_in(content_of(result)) == []
        assert str(content[1]) in file_text(result, "test_remote2")

    def test_small_remote_url_image_returns_exact_original(self):
        """A remote URL alone is no transformation, so nothing changes at all."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        runtime = make_runtime("test_remote3")

        content = [
            text_block("meta"),
            {"type": "image_url", "image_url": {"url": "https://example.com/c.png"}},
        ]
        message = ToolMessage(content=content, tool_call_id="test_remote3")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert result is message

    def test_image_order_preserved_labels_are_not(self):
        """#16 (§5.2.2): relative image order kept; inline labels honestly lost."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_order")

        img_a = image_url_block(30_000)
        img_b = anthropic_image_block(30_000)
        content = [
            text_block("Image 1: " + "pad " * 3000),
            img_a,
            text_block("Image 2: " + "pad " * 3000),
            img_b,
        ]
        message = ToolMessage(content=content, tool_call_id="test_order")
        result = middleware._intercept_large_tool_result(message, runtime)

        out = content_of(result)
        assert isinstance(out, list)
        images = image_blocks_in(out)
        assert len(images) == 2
        assert images[0]["type"] == "image_url"
        assert images[1]["type"] == "image"

        # The labels survive only inside the offload notice's preview of the
        # evicted text; they are NOT re-emitted as blocks adjacent to their
        # image, so the model cannot tell which image was "Image 1".
        assert "Tool result too large" in all_text_of(out)
        label_blocks = [
            block
            for block in out
            if isinstance(block, dict)
            and block.get("type") == "text"
            and "Tool result too large" not in str(block.get("text", ""))
        ]
        assert label_blocks == [], "labels must not be preserved as standalone blocks"

        # Exactly one text block remains: the notice, at the first evictable
        # position, i.e. ahead of both images.
        assert [b.get("type") for b in out] == ["text", "image_url", "image"]

    def test_command_wrapped_results_preserve_images_per_message(self):
        """#13: Command-wrapped multi-message results are handled per message."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_cmd_a")

        msg_a = ToolMessage(
            content=[text_block("A " * 5000), image_url_block(30_000)],
            tool_call_id="test_cmd_a",
        )
        msg_b = ToolMessage(
            content=[text_block("B " * 5000), anthropic_image_block(30_000)],
            tool_call_id="test_cmd_b",
        )
        command = Command(update={"messages": [msg_a, msg_b], "files": {}})
        result = middleware._intercept_large_tool_result(command, runtime)

        assert isinstance(result, Command)
        messages = result.update["messages"]
        assert len(messages) == 2
        for msg in messages:
            assert len(image_blocks_in(msg.content)) == 1


class TestPinnedEvictableRepresentation:
    """§5.2.1, test #17."""

    def test_single_text_block_special_case_survives(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_single")

        content = [text_block("Hello world! " * 1000)]
        message = ToolMessage(content=content, tool_call_id="test_single")
        result = middleware._intercept_large_tool_result(message, runtime)

        text = file_text(result, "test_single")
        assert text.startswith("Hello world!")
        assert not text.startswith("[{")

    def test_single_text_block_special_case_applies_beside_an_image(self):
        """One evictable text block plus a preserved image keeps the readable form."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_single_img")

        content = [text_block("Hello world! " * 1000), image_url_block(30_000)]
        message = ToolMessage(content=content, tool_call_id="test_single_img")
        result = middleware._intercept_large_tool_result(message, runtime)

        text = file_text(result, "test_single_img")
        assert text.startswith("Hello world!")
        assert PAYLOAD_MARKER not in text

    def test_stringified_list_form_byte_identical_for_image_free_content(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_pin")

        content = [text_block("First " * 500), {"type": "custom", "payload": "z" * 2000}]
        message = ToolMessage(content=content, tool_call_id="test_pin")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert file_text(result, "test_pin") == str(content)


class TestCapsWithoutOffload:
    """§5.1.1 — caps must materialize even when no offload happens (#21-#25)."""

    def test_oversized_image_downgraded_with_small_text(self):
        """#21: payload rejected, no file written, no offload notice."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_cap_size")

        content = [
            text_block("meta"),
            image_url_block(_MAX_PRESERVED_IMAGE_ENCODED_BYTES + 4),
        ]
        message = ToolMessage(content=content, tool_call_id="test_cap_size")
        result = middleware._intercept_large_tool_result(message, runtime)

        out = content_of(result)
        assert image_blocks_in(out) == []
        assert PAYLOAD_MARKER not in all_text_of(out)
        assert "Image omitted" in all_text_of(out)
        assert "Tool result too large" not in all_text_of(out)
        assert files_of(result) == {}

    @pytest.mark.parametrize(
        "bad_block",
        [
            {"type": "image_url"},
            {"type": "image_url", "image_url": {}},
            {"type": "image"},
            {"type": "image", "source": {}},
            {"type": "image", "data": ""},
        ],
    )
    def test_malformed_image_blocks_never_raise(self, bad_block):
        """#22 / #10: malformed blocks must neither crash nor vanish."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_bad")

        message = ToolMessage(content=[text_block("meta"), bad_block], tool_call_id="test_bad")
        result = middleware._intercept_large_tool_result(message, runtime)

        rendered = str(content_of(result))
        assert rendered  # produced something, did not raise
        assert image_blocks_in(content_of(result)) == []

    def test_bare_string_image_url_is_treated_as_a_remote_reference(self):
        """A bare-string `image_url` is a URL, not a malformed inline payload.

        It carries no data URI, so there is nothing to preserve, and deepagents
        must leave it untouched rather than declare it unusable.
        """
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        runtime = make_runtime("test_barestr")

        block = {"type": "image_url", "image_url": "not-a-dict"}
        message = ToolMessage(content=[text_block("meta"), block], tool_call_id="test_barestr")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert result is message

    def test_block_count_cap_downgrades_excess(self):
        """#23 / #9: more than the count cap → excess downgraded."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_count")

        per_block = _MAX_PRESERVED_IMAGE_TOTAL_BYTES // (_MAX_PRESERVED_IMAGE_BLOCKS + 4) // 4 * 4
        content = [text_block("meta")] + [
            image_url_block(per_block) for _ in range(_MAX_PRESERVED_IMAGE_BLOCKS + 3)
        ]
        message = ToolMessage(content=content, tool_call_id="test_count")
        result = middleware._intercept_large_tool_result(message, runtime)

        out = content_of(result)
        assert len(image_blocks_in(out)) == _MAX_PRESERVED_IMAGE_BLOCKS
        assert "Image omitted" in all_text_of(out)

    def test_aggregate_budget_preserves_deterministic_prefix(self):
        """#24 / #15: aggregate budget honoured, prefix deterministic."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_agg")

        per_block = (_MAX_PRESERVED_IMAGE_TOTAL_BYTES // 3 + 4) // 4 * 4
        content = [text_block("meta")] + [image_url_block(per_block) for _ in range(4)]
        message = ToolMessage(content=content, tool_call_id="test_agg")
        result = middleware._intercept_large_tool_result(message, runtime)

        out = content_of(result)
        assert len(image_blocks_in(out)) == 2
        assert "Image omitted" in all_text_of(out)

    def test_aggregate_budget_exact_boundary(self):
        """#15: exactly at budget preserved; one byte over downgraded."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)

        exact = ToolMessage(
            content=[text_block("meta"), image_url_block(_MAX_PRESERVED_IMAGE_TOTAL_BYTES)],
            tool_call_id="test_exact",
        )
        result_exact = middleware._intercept_large_tool_result(exact, make_runtime("test_exact"))
        assert len(image_blocks_in(content_of(result_exact))) == 1

        over = ToolMessage(
            content=[text_block("meta"), image_url_block(_MAX_PRESERVED_IMAGE_TOTAL_BYTES + 4)],
            tool_call_id="test_over",
        )
        result_over = middleware._intercept_large_tool_result(over, make_runtime("test_over"))
        assert image_blocks_in(content_of(result_over)) == []

    def test_noop_classification_returns_exact_original_object(self):
        """#25: the `return original` branch is reachable only on a no-op."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_noop")

        content = [text_block("meta"), image_url_block(1000)]
        message = ToolMessage(content=content, tool_call_id="test_noop")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert result is message


class TestInlinePayloadValidation:
    """The encoded-length precheck must reject anything that is not base64."""

    @pytest.mark.parametrize(
        ("label", "url"),
        [
            # A valid data URI that is NOT base64: measuring it as an encoded
            # image payload would be meaningless.
            ("plain_text_data_uri", "data:image/png,hello"),
            ("no_base64_marker", "data:image/png;charset=utf-8,QUJD"),
            ("non_ascii_payload", "data:image/png;base64,QUJDéééé"),
            ("bad_alphabet", "data:image/png;base64,QUJD!!!!"),
            ("bad_length", "data:image/png;base64,QUJDR"),
            ("too_much_padding", "data:image/png;base64,QQ==="),
            ("empty_payload", "data:image/png;base64,"),
        ],
    )
    def test_non_base64_inline_payloads_are_downgraded(self, label, url):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        tool_call_id = f"test_badb64_{label}"
        runtime = make_runtime(tool_call_id)

        content = [text_block("meta"), {"type": "image_url", "image_url": {"url": url}}]
        message = ToolMessage(content=content, tool_call_id=tool_call_id)
        result = middleware._intercept_large_tool_result(message, runtime)

        out = content_of(result)
        assert image_blocks_in(out) == [], f"{label} was wrongly preserved as an image"
        assert "Image omitted" in all_text_of(out)

    @pytest.mark.parametrize(
        ("label", "block"),
        [
            ("mcp_non_ascii", {"type": "image", "data": "QUJDéééé"}),
            ("mcp_bad_alphabet", {"type": "image", "data": "QUJD****"}),
            ("anthropic_non_ascii", {"type": "image", "source": {"data": "QUJDéééé"}}),
            ("anthropic_bad_alphabet", {"type": "image", "source": {"data": "QUJD****"}}),
        ],
    )
    def test_non_base64_alternate_shapes_are_downgraded(self, label, block):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        tool_call_id = f"test_badshape_{label}"
        runtime = make_runtime(tool_call_id)

        message = ToolMessage(content=[text_block("meta"), block], tool_call_id=tool_call_id)
        result = middleware._intercept_large_tool_result(message, runtime)

        assert image_blocks_in(content_of(result)) == []
        assert "Image omitted" in all_text_of(content_of(result))

    def test_valid_padded_base64_is_accepted(self):
        """Padding is legal base64 and must not be rejected."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=1000)
        runtime = make_runtime("test_padded")

        content = [
            text_block("meta"),
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
        ]
        message = ToolMessage(content=content, tool_call_id="test_padded")
        result = middleware._intercept_large_tool_result(message, runtime)

        assert result is message


class TestAggregateBudgetLatch:
    """Once the aggregate budget is exhausted, every later image is downgraded."""

    def test_later_small_image_cannot_slip_through(self):
        """A large over-budget image must not let a tiny later one be preserved.

        This is the case an equal-size test cannot catch: without a latch the
        preserved set would depend on block sizes rather than on position.
        """
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_latch")

        half = _MAX_PRESERVED_IMAGE_TOTAL_BYTES // 2 // 4 * 4
        content = [
            text_block("meta"),
            image_url_block(half),  # fits, leaving ~half the budget free
            image_url_block(_MAX_PRESERVED_IMAGE_TOTAL_BYTES),  # first over-budget block
            image_url_block(64),  # tiny: WOULD fit the remaining budget
        ]
        message = ToolMessage(content=content, tool_call_id="test_latch")
        result = middleware._intercept_large_tool_result(message, runtime)

        out = content_of(result)
        preserved = image_blocks_in(out)
        # Without the latch the 64-char image would also be preserved, giving 2.
        assert len(preserved) == 1, "budget exhaustion must latch for all later images"
        assert all_text_of(out).count("aggregate image budget") == 2


class TestWriteFailureDoesNotResurrectPayloads:
    """A failed offload must not undo the classification decisions."""

    class _FailingBackend:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def write(self, *args, **kwargs):
            return type("R", (), {"error": "disk full", "files_update": None})()

        async def awrite(self, *args, **kwargs):
            return type("R", (), {"error": "disk full", "files_update": None})()

    def _oversized_with_large_text(self):
        return [
            text_block("LOG " * 5000),
            image_url_block(_MAX_PRESERVED_IMAGE_ENCODED_BYTES + 4),
        ]

    def test_sync_write_failure_keeps_rejection(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_wfail")
        backend = self._FailingBackend(middleware._get_backend(runtime))

        content = self._oversized_with_large_text()
        message = ToolMessage(content=content, tool_call_id="test_wfail")
        processed, files_update = middleware._process_large_message(message, backend)

        assert files_update is None
        assert PAYLOAD_MARKER not in all_text_of(processed.content)
        assert image_blocks_in(processed.content) == []
        assert "Image omitted" in all_text_of(processed.content)
        # The evictable text could not be offloaded, so it must remain.
        assert "LOG" in all_text_of(processed.content)

    @pytest.mark.asyncio
    async def test_async_write_failure_keeps_rejection(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_wfail_a")
        backend = self._FailingBackend(middleware._get_backend(runtime))

        content = self._oversized_with_large_text()
        message = ToolMessage(content=content, tool_call_id="test_wfail_a")
        processed, files_update = await middleware._aprocess_large_message(message, backend)

        assert files_update is None
        assert PAYLOAD_MARKER not in all_text_of(processed.content)
        assert image_blocks_in(processed.content) == []
        assert "Image omitted" in all_text_of(processed.content)

    def test_write_failure_without_transformation_returns_original(self):
        """No classification change means the original object is still returned."""
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_wfail_noop")
        backend = self._FailingBackend(middleware._get_backend(runtime))

        message = ToolMessage(content="x" * 5000, tool_call_id="test_wfail_noop")
        processed, files_update = middleware._process_large_message(message, backend)

        assert processed is message
        assert files_update is None


class TestMessageMetadataIsPreserved:
    """Rebuilding a ToolMessage must not silently drop its other fields."""

    def _message(self, content, tool_call_id):
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name="example_tool",
            id="msg-abc-123",
            status="error",
            artifact={"kind": "diagnostic", "code": 42},
            response_metadata={"provider": "unit-test"},
        )

    def _assert_metadata_intact(self, msg):
        assert msg.id == "msg-abc-123"
        assert msg.name == "example_tool"
        assert msg.status == "error"
        assert msg.artifact == {"kind": "diagnostic", "code": 42}
        assert msg.response_metadata == {"provider": "unit-test"}

    def test_metadata_survives_cap_transformation_without_offload(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_meta_cap")

        content = [text_block("meta"), image_url_block(_MAX_PRESERVED_IMAGE_ENCODED_BYTES + 4)]
        result = middleware._intercept_large_tool_result(
            self._message(content, "test_meta_cap"), runtime
        )

        message = result.update["messages"][0] if isinstance(result, Command) else result
        self._assert_metadata_intact(message)

    def test_metadata_survives_eviction_with_preserved_image(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_meta_evict")

        content = [text_block("LOG " * 5000), image_url_block(30_000)]
        result = middleware._intercept_large_tool_result(
            self._message(content, "test_meta_evict"), runtime
        )

        assert isinstance(result, Command)
        self._assert_metadata_intact(result.update["messages"][0])

    def test_metadata_survives_plain_string_eviction(self):
        middleware = FilesystemMiddleware(tool_token_limit_before_evict=100)
        runtime = make_runtime("test_meta_str")

        result = middleware._intercept_large_tool_result(
            self._message("x" * 5000, "test_meta_str"), runtime
        )

        assert isinstance(result, Command)
        self._assert_metadata_intact(result.update["messages"][0])


class TestSyncAsyncParity:
    """#12: the async twin must behave identically."""

    CASES = [
        ("parity_read_image", [text_block("meta"), image_url_block(129_000)]),
        ("parity_text_only", [text_block("First " * 500), text_block("Second " * 500)]),
        ("parity_mix", [text_block("LOG " * 5000), image_url_block(50_000)]),
        ("parity_small", [text_block("small"), image_url_block(100)]),
        (
            "parity_oversize",
            [text_block("meta"), image_url_block(_MAX_PRESERVED_IMAGE_ENCODED_BYTES + 4)],
        ),
        ("parity_str", "x" * 5000),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("tool_call_id", "content"), CASES)
    async def test_sync_and_async_agree(self, tool_call_id, content):
        sync_mw = FilesystemMiddleware(tool_token_limit_before_evict=100)
        async_mw = FilesystemMiddleware(tool_token_limit_before_evict=100)

        sync_msg = ToolMessage(content=content, tool_call_id=tool_call_id)
        async_msg = ToolMessage(content=content, tool_call_id=tool_call_id)

        sync_out = sync_mw._process_large_message(sync_msg, sync_mw._get_backend(make_runtime(tool_call_id)))
        async_out = await async_mw._aprocess_large_message(
            async_msg, async_mw._get_backend(make_runtime(tool_call_id))
        )

        assert sync_out[0].content == async_out[0].content
        sync_files = sync_out[1] or {}
        async_files = async_out[1] or {}
        assert sorted(sync_files) == sorted(async_files)
        for path in sync_files:
            assert sync_files[path]["content"] == async_files[path]["content"]


class TestBase64SanityOfHelpers:
    """The test helpers must produce genuinely base64-shaped payloads."""

    def test_data_url_body_is_valid_base64_alphabet(self):
        url = data_url(64)
        body = url.split(",", 1)[1]
        assert len(body) == 64
        base64.b64decode(body + "=" * (-len(body) % 4))
