from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from telegram import ReplyKeyboardMarkup

from llm_tg_bot.commands import CommandHandler
from llm_tg_bot.config import Settings
from llm_tg_bot.providers import PreparedRequest, ProviderAdapter, ProviderSpec
from llm_tg_bot.rendering import OutgoingMessage
from llm_tg_bot.session import SessionManager


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


class CommandHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_interactive_new_directory_choice_does_not_restore_control_keyboard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            handler, sent_messages = self._build_handler(Path(tempdir))

            await handler.handle(9, "/new")
            await handler.handle_pending_input(9, "fake")
            await handler.handle_pending_input(9, ".")

        self.assertEqual(len(sent_messages), 3)
        self.assertIsNotNone(sent_messages[0][2])
        self.assertIsNotNone(sent_messages[1][2])
        self.assertIsNone(sent_messages[2][2])
        self.assertIn("[session started: fake | workdir=", sent_messages[2][1])

    async def test_direct_new_with_directory_keeps_control_keyboard(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            handler, sent_messages = self._build_handler(Path(tempdir))

            await handler.handle(9, "/new fake .")

        self.assertEqual(len(sent_messages), 1)
        self.assertIsNotNone(sent_messages[0][2])
        self.assertIn("[session started: fake | workdir=", sent_messages[0][1])

    async def test_queue_command_shows_empty_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            handler, sent_messages = self._build_handler(Path(tempdir))

            await handler.handle(9, "/new fake .")
            await handler.handle(9, "/queue")

        self.assertEqual(len(sent_messages), 2)
        self.assertIn("Queue is empty", sent_messages[1][1])

    async def test_queue_command_shows_queued_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            handler, sent_messages = self._build_handler(Path(tempdir))

            await handler.handle(9, "/new fake .")
            # Get session manager from handler and directly add to queue
            session_manager = handler._session_manager
            record = session_manager._records.get(9)
            self.assertIsNotNone(record)
            # Directly add prompts to the queue without triggering execution
            record.pending_prompts.append("First prompt")
            record.pending_prompts.append("Second prompt")
            await handler.handle(9, "/queue")

        self.assertGreaterEqual(len(sent_messages), 2)
        queue_message = sent_messages[-1][1]
        self.assertIn("Queue (2 item(s)):", queue_message)
        self.assertIn("1. First prompt", queue_message)
        self.assertIn("2. Second prompt", queue_message)

    def _build_handler(
        self,
        workdir: Path,
    ) -> tuple[
        CommandHandler,
        list[tuple[int, str, ReplyKeyboardMarkup | None]],
    ]:
        settings = Settings(
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
            providers={"fake": ProviderSpec(adapter=_FakeAdapter(), cwd=workdir)},
        )
        sent_messages: list[tuple[int, str, ReplyKeyboardMarkup | None]] = []

        async def output_callback(chat_id: int, message: OutgoingMessage) -> None:
            del chat_id, message

        async def send_message(
            chat_id: int,
            text: str,
            reply_markup: ReplyKeyboardMarkup | None = None,
        ) -> None:
            sent_messages.append((chat_id, text, reply_markup))

        manager = SessionManager(
            providers=settings.providers,
            idle_timeout_seconds=settings.session_idle_timeout_seconds,
            output_callback=output_callback,
        )
        handler = CommandHandler(
            settings=settings,
            session_manager=manager,
            send_message=send_message,
            keyboard_factory=lambda: ReplyKeyboardMarkup([["/new"]]),
        )
        return handler, sent_messages
