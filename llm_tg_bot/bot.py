from __future__ import annotations

import asyncio
import contextlib
import logging

from telegram import Bot, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import RetryAfter, TelegramError

from llm_tg_bot.commands import CommandHandler, is_bot_command
from llm_tg_bot.config import Settings
from llm_tg_bot.rendering import (
    OutgoingMessage,
    RenderMode,
    RenderedChunk,
    build_message_chunks,
)
from llm_tg_bot.session import SessionManager

logger = logging.getLogger(__name__)

_CONTROL_KEYBOARD_LAYOUT = [
    ["/new", "/status", "/cancel"],
    ["/list", "/stop", "/help"],
]
_TYPING_ACTION_INTERVAL_SECONDS = 4.0


class BridgeBot:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bot = Bot(token=settings.telegram_bot_token)
        self._session_manager = SessionManager(
            providers=settings.providers,
            idle_timeout_seconds=settings.session_idle_timeout_seconds,
            output_callback=self._send_output,
            request_started_callback=self._start_typing_indicator,
        )
        self._command_handler = CommandHandler(
            settings=settings,
            session_manager=self._session_manager,
            send_message=self._send_message,
            keyboard_factory=_control_keyboard,
        )
        self._send_locks: dict[int, asyncio.Lock] = {}
        self._typing_tasks: dict[int, asyncio.Task[None]] = {}
        self._offset: int | None = None
        self._idle_cleanup_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        self._idle_cleanup_task = asyncio.create_task(self._idle_cleanup_loop())
        try:
            await self._bot.initialize()
            while True:
                try:
                    request_kwargs: dict[str, int] = {
                        "timeout": self._settings.poll_timeout_seconds,
                    }
                    if self._offset is not None:
                        request_kwargs["offset"] = self._offset
                    updates = await self._bot.get_updates(**request_kwargs)
                except RetryAfter as exc:
                    delay = _retry_after_seconds(exc)
                    logger.warning(
                        "Rate limited while polling Telegram; retrying in %.2fs",
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                except TelegramError as exc:
                    logger.warning("Telegram polling failed: %s", exc)
                    await asyncio.sleep(3)
                    continue

                for update in updates:
                    self._offset = update.update_id + 1
                    try:
                        await self._handle_update(update)
                    except Exception:
                        logger.exception("Failed to handle update: %s", update)
        finally:
            if self._idle_cleanup_task:
                self._idle_cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._idle_cleanup_task
            await self._stop_all_typing_tasks()
            with contextlib.suppress(TelegramError):
                await self._bot.shutdown()

    async def _handle_update(self, update: Update) -> None:
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not message or not chat:
            return

        raw_text = message.text or ""
        if not raw_text.strip():
            return

        chat_id = int(chat.id)
        user_id = int(user.id) if user else None
        text = raw_text.strip()

        if not self._is_allowed_user(user_id):
            await self._send_message(chat_id, "Access denied for this user.")
            logger.warning(
                "Denied user_id=%s chat_id=%s",
                user_id if user_id is not None else "<none>",
                chat_id,
            )
            return

        if self._command_handler.has_pending_new_session(chat_id):
            if not (text.startswith("/") and is_bot_command(text)):
                handled = await self._command_handler.handle_pending_input(
                    chat_id, raw_text
                )
                if handled:
                    return

        if text.startswith("/"):
            if is_bot_command(text):
                try:
                    await self._command_handler.handle(chat_id, text)
                except ValueError as exc:
                    await self._send_message(chat_id, f"Error: {exc}")
            else:
                await self._forward_text(chat_id, text)
            return

        await self._forward_text(chat_id, raw_text)

    def _active_or_default_provider(self, chat_id: int) -> str:
        return self._session_manager.active_provider_name(
            chat_id
        ) or self._command_handler.preferred_provider(chat_id)

    async def _forward_text(self, chat_id: int, text: str) -> None:
        provider_name = self._active_or_default_provider(chat_id)
        had_session = self._session_manager.has_session(chat_id)
        try:
            send_result = await self._session_manager.send_text(
                chat_id, text, provider_name
            )
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            logger.warning("Failed to deliver input for chat_id=%s: %s", chat_id, exc)
            await self._send_message(chat_id, f"Failed to send prompt: {exc}")
            return

        record = send_result.record
        if send_result.queued_ahead > 0:
            await self._send_message(
                chat_id,
                f"[request queued: {send_result.queued_ahead} ahead]",
            )

        if not had_session:
            workdir = record.provider.cwd or "(current working directory)"
            await self._send_message(
                chat_id,
                f"[session started: {record.provider.name} | workdir={workdir}]",
                reply_markup=_control_keyboard(),
            )

    def _is_allowed_user(self, user_id: int | None) -> bool:
        if self._settings.allow_all_users:
            return True
        return user_id is not None and user_id in self._settings.allowed_user_ids

    async def _send_output(self, chat_id: int, message: OutgoingMessage) -> None:
        try:
            await self._send_message(
                chat_id,
                message.text,
                render_mode=message.render_mode,
            )
        except TelegramError as exc:
            logger.warning("Failed to send output for chat_id=%s: %s", chat_id, exc)

    def _start_typing_indicator(
        self,
        chat_id: int,
        request_task: asyncio.Task[None],
    ) -> None:
        self._cancel_typing_indicator(chat_id)
        typing_task = asyncio.create_task(self._typing_loop(chat_id, request_task))
        self._typing_tasks[chat_id] = typing_task
        request_task.add_done_callback(lambda _: typing_task.cancel())
        typing_task.add_done_callback(
            lambda _: self._clear_typing_indicator(chat_id, typing_task)
        )

    def _cancel_typing_indicator(self, chat_id: int) -> None:
        typing_task = self._typing_tasks.pop(chat_id, None)
        if typing_task:
            typing_task.cancel()

    def _clear_typing_indicator(
        self,
        chat_id: int,
        typing_task: asyncio.Task[None],
    ) -> None:
        if self._typing_tasks.get(chat_id) is typing_task:
            self._typing_tasks.pop(chat_id, None)

    async def _stop_all_typing_tasks(self) -> None:
        typing_tasks = list(self._typing_tasks.values())
        self._typing_tasks.clear()
        for task in typing_tasks:
            task.cancel()
        for task in typing_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _typing_loop(
        self,
        chat_id: int,
        request_task: asyncio.Task[None],
    ) -> None:
        try:
            if not await self._send_chat_action(chat_id, ChatAction.TYPING):
                return
            while not request_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(request_task),
                        timeout=_TYPING_ACTION_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if not await self._send_chat_action(chat_id, ChatAction.TYPING):
                        return
        except asyncio.CancelledError:
            raise

    async def _send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: ReplyKeyboardMarkup | None = None,
        *,
        render_mode: RenderMode = RenderMode.PLAIN,
    ) -> None:
        if not text:
            return

        lock = self._send_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            chunks = build_message_chunks(
                OutgoingMessage(text, render_mode=render_mode),
                self._settings.message_max_chars,
            )
            last_index = len(chunks) - 1
            for index, chunk in enumerate(chunks):
                await self._send_chunk(
                    chat_id,
                    chunk,
                    reply_markup=reply_markup if index == last_index else None,
                )

    async def _send_chunk(
        self,
        chat_id: int,
        chunk: RenderedChunk,
        reply_markup: ReplyKeyboardMarkup | None = None,
    ) -> None:
        text = chunk.text
        parse_mode = chunk.parse_mode
        while True:
            try:
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                )
                return
            except RetryAfter as exc:
                delay = _retry_after_seconds(exc)
                logger.warning(
                    "Rate limited while sending to chat_id=%s; retrying in %.2fs",
                    chat_id,
                    delay,
                )
                await asyncio.sleep(delay)
            except TelegramError:
                if parse_mode is None:
                    raise
                logger.warning(
                    "Formatted send failed for chat_id=%s; retrying as plain text",
                    chat_id,
                )
                text = chunk.plain_text
                parse_mode = None

    async def _send_chat_action(self, chat_id: int, action: ChatAction) -> bool:
        while True:
            try:
                await self._bot.send_chat_action(chat_id=chat_id, action=action)
                return True
            except RetryAfter as exc:
                delay = _retry_after_seconds(exc)
                logger.warning(
                    "Rate limited while sending chat action to chat_id=%s; retrying in %.2fs",
                    chat_id,
                    delay,
                )
                await asyncio.sleep(delay)
            except TelegramError as exc:
                logger.warning(
                    "Failed to send chat action for chat_id=%s: %s",
                    chat_id,
                    exc,
                )
                return False

    async def _idle_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            await self._session_manager.stop_idle_sessions()


def _retry_after_seconds(exc: RetryAfter) -> float:
    retry_after = exc.retry_after
    if hasattr(retry_after, "total_seconds"):
        return max(0.0, float(retry_after.total_seconds()))
    return max(0.0, float(retry_after))


def _control_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        _CONTROL_KEYBOARD_LAYOUT,
        resize_keyboard=True,
        input_field_placeholder="Send text or tap a control key",
    )
