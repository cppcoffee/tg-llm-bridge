from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

from llm_tg_bot.bot import BridgeBot
from llm_tg_bot.config import Settings
from llm_tg_bot.providers import PreparedRequest, ProviderAdapter, ProviderSpec
from llm_tg_bot.rendering import OutgoingMessage


class _FakeAdapter(ProviderAdapter):
    name = "fake"
    executable = "fake"

    def prepare_request(
        self,
        prompt: str,
        resume: bool,
        *,
        skip_git_repo_check: bool = False,
    ) -> PreparedRequest:
        del prompt, resume, skip_git_repo_check
        return PreparedRequest(command=(self.executable,))

    def build_response(
        self,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
        output_file: Path | None,
    ) -> str:
        del stderr_text, return_code, output_file
        return stdout_text


class BridgeBotTests(unittest.IsolatedAsyncioTestCase):
    def test_bridge_bot_uses_separate_requests_for_polling_and_outbound_sends(
        self,
    ) -> None:
        settings = self._build_settings()
        outbound_request = object()
        polling_request = object()
        bot_instance = MagicMock()

        with patch(
            "llm_tg_bot.bot.HTTPXRequest",
            side_effect=[outbound_request, polling_request],
        ) as request_class, patch(
            "llm_tg_bot.bot.Bot",
            return_value=bot_instance,
        ) as bot_class:
            BridgeBot(settings)

        self.assertEqual(
            request_class.call_args_list,
            [
                call(connection_pool_size=8, pool_timeout=5.0),
                call(connection_pool_size=1, pool_timeout=5.0),
            ],
        )
        bot_class.assert_called_once_with(
            token="token",
            request=outbound_request,
            get_updates_request=polling_request,
        )

    async def test_send_output_stops_typing_indicator_before_sending(self) -> None:
        settings = self._build_settings()

        with patch("llm_tg_bot.bot.HTTPXRequest"), patch("llm_tg_bot.bot.Bot"):
            bot = BridgeBot(settings)

        typing_stopped = asyncio.Event()
        block_forever = asyncio.Event()

        async def typing_task() -> None:
            try:
                await block_forever.wait()
            except asyncio.CancelledError:
                typing_stopped.set()
                raise

        task = asyncio.create_task(typing_task())
        bot._typing_tasks[9] = task
        await asyncio.sleep(0)

        async def fake_send_message(
            chat_id: int,
            text: str,
            reply_markup=None,
            *,
            render_mode,
        ) -> None:
            del reply_markup
            self.assertEqual(chat_id, 9)
            self.assertEqual(text, "hello")
            self.assertTrue(typing_stopped.is_set())

        bot._send_message = AsyncMock(side_effect=fake_send_message)

        await bot._send_output(9, OutgoingMessage("hello"))

        self.assertNotIn(9, bot._typing_tasks)
        self.assertTrue(task.cancelled())
        bot._send_message.assert_awaited_once()

    @staticmethod
    def _build_settings() -> Settings:
        return Settings(
            telegram_bot_token="token",
            allow_all_users=True,
            allowed_user_ids=frozenset(),
            default_provider="fake",
            poll_timeout_seconds=30,
            telegram_connection_pool_size=8,
            telegram_pool_timeout_seconds=5.0,
            message_max_chars=4000,
            session_idle_timeout_seconds=2700,
            log_level="INFO",
            providers={"fake": ProviderSpec(adapter=_FakeAdapter())},
        )
