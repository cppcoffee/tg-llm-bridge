from __future__ import annotations

import unittest

from llm_tg_bot.rendering import OutgoingMessage, RenderMode, build_message_chunks


class RenderingTests(unittest.TestCase):
    def test_markdown_chunks_render_common_constructs(self) -> None:
        chunks = build_message_chunks(
            OutgoingMessage(
                "# Title\n\n- **bold** item\n\n`inline`\n\n```python\nprint('hi')\n```",
                render_mode=RenderMode.MARKDOWN,
            ),
            limit=4096,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].parse_mode, "HTML")
        self.assertIn("<b>Title</b>", chunks[0].text)
        self.assertIn("• <b>bold</b> item", chunks[0].text)
        self.assertIn("<code>inline</code>", chunks[0].text)
        self.assertIn("<pre>print(&#x27;hi&#x27;)</pre>", chunks[0].text)

    def test_markdown_chunks_split_code_blocks_without_breaking_tags(self) -> None:
        chunks = build_message_chunks(
            OutgoingMessage(
                "```\nline 1\nline 2\nline 3\nline 4\n```",
                render_mode=RenderMode.MARKDOWN,
            ),
            limit=24,
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.parse_mode, "HTML")
            self.assertTrue(chunk.text.startswith("<pre>"))
            self.assertTrue(chunk.text.endswith("</pre>"))

    def test_plain_text_chunks_do_not_set_parse_mode(self) -> None:
        chunks = build_message_chunks(
            OutgoingMessage("[session started]"),
            limit=4096,
        )

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].parse_mode)
        self.assertEqual(chunks[0].text, "[session started]")
