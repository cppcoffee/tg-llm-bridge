from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape

from telegram.constants import ParseMode

_RAW_HEADROOM = 512
_CODE_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_UNORDERED_LIST_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED_LIST_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^(\s*>+)\s?(.*)$")
_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ALT_BOLD_RE = re.compile(r"__(.+?)__")


@dataclass(frozen=True, slots=True)
class PreparedMessageChunk:
    plain_text: str
    formatted_text: str
    parse_mode: str | None


def prepare_message_chunks(text: str, limit: int) -> list[PreparedMessageChunk]:
    if not text:
        return []

    headroom = min(_RAW_HEADROOM, max(0, limit // 5))
    raw_limit = max(1, limit - headroom)
    prepared: list[PreparedMessageChunk] = []
    for raw_chunk in _split_markdown_text(text, raw_limit):
        prepared.extend(_prepare_chunk(raw_chunk, limit))
    return prepared


def _prepare_chunk(text: str, limit: int) -> list[PreparedMessageChunk]:
    formatted_text = _render_markdown_as_html(text)
    if len(formatted_text) <= limit or len(text) <= 1:
        return [
            PreparedMessageChunk(
                plain_text=text,
                formatted_text=formatted_text,
                parse_mode=ParseMode.HTML,
            )
        ]

    next_limit = max(1, min(len(text) - 1, len(text) // 2))
    chunks = _split_markdown_text(text, next_limit)
    if len(chunks) == 1 and chunks[0] == text:
        return [
            PreparedMessageChunk(
                plain_text=segment,
                formatted_text=segment,
                parse_mode=None,
            )
            for segment in _split_plain_text(text, limit)
        ]

    prepared: list[PreparedMessageChunk] = []
    for chunk in chunks:
        prepared.extend(_prepare_chunk(chunk, limit))
    return prepared


def _split_markdown_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _CODE_FENCE_RE.match(line):
            end_index = _find_closing_fence(lines, index + 1)
            if end_index is not None:
                block = "\n".join(lines[index : end_index + 1])
                blocks.extend(_split_code_block(block, limit))
                index = end_index + 1
                continue

        blocks.extend(_split_plain_line(line, limit))
        index += 1

    return _pack_blocks(blocks, limit)


def _find_closing_fence(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if _CODE_FENCE_RE.match(lines[index]):
            return index
    return None


def _split_code_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]

    lines = block.splitlines()
    opening = lines[0]
    closing = lines[-1]
    code_lines = lines[1:-1]

    overhead = len(opening) + len(closing) + 2
    content_limit = max(1, limit - overhead)

    content_blocks: list[str] = []
    pending_lines: list[str] = []
    pending_length = 0

    for line in code_lines:
        for segment in _split_plain_line(line, content_limit):
            separator = 1 if pending_lines else 0
            if pending_lines and pending_length + separator + len(segment) > content_limit:
                content_blocks.append("\n".join(pending_lines))
                pending_lines = [segment]
                pending_length = len(segment)
                continue

            pending_lines.append(segment)
            pending_length += separator + len(segment)

    if pending_lines:
        content_blocks.append("\n".join(pending_lines))

    if not content_blocks:
        return [f"{opening}\n{closing}"]

    return [f"{opening}\n{content}\n{closing}" for content in content_blocks]


def _split_plain_line(line: str, limit: int) -> list[str]:
    if len(line) <= limit:
        return [line]

    segments: list[str] = []
    remaining = line
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        segments.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining or not segments:
        segments.append(remaining)
    return segments


def _split_plain_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            newline_index = text.rfind("\n", start, end)
            if newline_index > start:
                end = newline_index + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def _pack_blocks(blocks: list[str], limit: int) -> list[str]:
    chunks: list[str] = []
    pending_lines: list[str] = []
    pending_length = 0

    for block in blocks:
        separator = 1 if pending_lines else 0
        block_length = len(block)
        if pending_lines and pending_length + separator + block_length > limit:
            chunks.append("\n".join(pending_lines))
            pending_lines = [block]
            pending_length = block_length
            continue

        pending_lines.append(block)
        pending_length += separator + block_length

    if pending_lines:
        chunks.append("\n".join(pending_lines))

    return chunks or [""]


def _render_markdown_as_html(text: str) -> str:
    lines = text.splitlines()
    rendered: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if _CODE_FENCE_RE.match(line):
            end_index = _find_closing_fence(lines, index + 1)
            if end_index is not None:
                code_content = "\n".join(lines[index + 1 : end_index])
                rendered.append(f"<pre>{escape(code_content, quote=False)}</pre>")
                index = end_index + 1
                continue

        rendered.append(_render_line(line))
        index += 1

    return "\n".join(rendered)


def _render_line(line: str) -> str:
    if not line:
        return ""

    heading_match = _HEADING_RE.match(line)
    if heading_match:
        return f"<b>{_render_inline(heading_match.group(1))}</b>"

    if _RULE_RE.match(line):
        return "----------"

    unordered_match = _UNORDERED_LIST_RE.match(line)
    if unordered_match:
        indent = _html_indent(unordered_match.group(1))
        return f"{indent}• {_render_inline(unordered_match.group(2))}"

    ordered_match = _ORDERED_LIST_RE.match(line)
    if ordered_match:
        indent = _html_indent(ordered_match.group(1))
        return f"{indent}{ordered_match.group(2)}. {_render_inline(ordered_match.group(3))}"

    quote_match = _BLOCKQUOTE_RE.match(line)
    if quote_match:
        quote_depth = quote_match.group(1).count(">")
        return f"{'&gt; ' * quote_depth}{_render_inline(quote_match.group(2))}"

    return _render_inline(line)


def _html_indent(indent: str) -> str:
    width = len(indent.replace("\t", "  ")) // 2
    return "&nbsp;&nbsp;" * max(0, width)


def _render_inline(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in _CODE_SPAN_RE.finditer(text):
        parts.append(_render_text_without_code(text[cursor : match.start()]))
        parts.append(f"<code>{escape(match.group(1), quote=False)}</code>")
        cursor = match.end()
    parts.append(_render_text_without_code(text[cursor:]))
    return "".join(parts)


def _render_text_without_code(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in _LINK_RE.finditer(text):
        parts.append(_apply_basic_formatting(text[cursor : match.start()]))
        label = _apply_basic_formatting(match.group(1))
        href = escape(match.group(2), quote=True)
        parts.append(f'<a href="{href}">{label}</a>')
        cursor = match.end()
    parts.append(_apply_basic_formatting(text[cursor:]))
    return "".join(parts)


def _apply_basic_formatting(text: str) -> str:
    escaped = escape(text, quote=False)
    escaped = _STRIKE_RE.sub(r"<s>\1</s>", escaped)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    escaped = _ALT_BOLD_RE.sub(r"<b>\1</b>", escaped)
    return escaped
