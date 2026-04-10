from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from telegram import ReplyKeyboardMarkup

from llm_tg_bot.config import Settings
from llm_tg_bot.session import SessionManager
from llm_tg_bot.workdirs import (
    directory_choices,
    directory_prompt,
    providers_text,
    resolve_workdir_choice,
)

SendMessage = Callable[[int, str, ReplyKeyboardMarkup | None], Awaitable[None]]
KeyboardFactory = Callable[[], ReplyKeyboardMarkup]
CommandAction = Callable[[int, str], Awaitable[None]]

_DIRECTORY_BUTTON_LIMIT = 24
_KEYBOARD_COLUMNS = 2


def command_name(text: str) -> str:
    return text.split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()


@dataclass(slots=True)
class PendingNewSession:
    provider_name: str | None = None


class CommandHandler:
    def __init__(
        self,
        settings: Settings,
        session_manager: SessionManager,
        send_message: SendMessage,
        keyboard_factory: KeyboardFactory,
    ) -> None:
        self._settings = settings
        self._session_manager = session_manager
        self._send_message = send_message
        self._keyboard_factory = keyboard_factory
        self._preferred_provider_by_chat: dict[int, str] = {}
        self._pending_new_session_by_chat: dict[int, PendingNewSession] = {}
        self._command_handlers: dict[str, CommandAction] = {
            "/help": self._handle_help,
            "/list": self._handle_list,
            "/use": self._handle_use,
            "/new": self._handle_new,
            "/status": self._handle_status,
            "/queue": self._handle_queue,
            "/stop": self._handle_stop,
            "/cancel": self._handle_cancel,
        }

    async def handle(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        command = command_name(parts[0])
        raw_arg = parts[1].strip() if len(parts) > 1 else ""
        handler = self._command_handlers.get(command)
        if handler is None:
            await self._send_message(chat_id, "Unknown command. Use /help.")
            return
        await handler(chat_id, raw_arg)

    def has_pending_new_session(self, chat_id: int) -> bool:
        return chat_id in self._pending_new_session_by_chat

    def is_command(self, text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and command_name(stripped) in self._command_handlers

    async def handle_pending_input(self, chat_id: int, text: str) -> bool:
        pending = self._pending_new_session_by_chat.get(chat_id)
        if pending is None:
            return False

        choice = text.strip()
        if pending.provider_name is None:
            await self._handle_pending_provider_choice(chat_id, pending, choice)
            return True

        await self._handle_pending_directory_choice(chat_id, pending.provider_name, choice)
        return True

    def preferred_provider(self, chat_id: int) -> str:
        return self._preferred_provider_by_chat.get(
            chat_id, self._settings.default_provider
        )

    async def _handle_help(self, chat_id: int, raw_arg: str) -> None:
        del raw_arg
        await self._send_message(
            chat_id,
            self._help_text(chat_id),
            reply_markup=self._keyboard_factory(),
        )

    async def _handle_list(self, chat_id: int, raw_arg: str) -> None:
        del raw_arg
        await self._send_message(chat_id, providers_text(self._settings.providers))

    async def _handle_use(self, chat_id: int, raw_arg: str) -> None:
        if not raw_arg:
            await self._send_message(chat_id, "Usage: /use <provider>")
            return
        await self._set_preferred_provider(chat_id, raw_arg.lower())

    async def _handle_new(self, chat_id: int, raw_arg: str) -> None:
        if not raw_arg:
            await self._begin_new_session(chat_id)
            return

        provider_name, directory_choice = self._parse_new_arguments(chat_id, raw_arg)
        if directory_choice is None:
            await self._begin_new_session(chat_id, provider_name=provider_name)
            return

        workdir = self._resolve_workdir_choice(provider_name, directory_choice)
        await self._start_session(chat_id, provider_name, workdir)

    async def _handle_status(self, chat_id: int, raw_arg: str) -> None:
        del raw_arg
        preferred = self.preferred_provider(chat_id)
        status = self._session_manager.status_text(chat_id)
        await self._send_message(chat_id, f"Preferred provider: {preferred}\n{status}")

    async def _handle_queue(self, chat_id: int, raw_arg: str) -> None:
        del raw_arg
        queue = self._session_manager.queue_text(chat_id)
        await self._send_message(chat_id, queue)

    async def _handle_stop(self, chat_id: int, raw_arg: str) -> None:
        del raw_arg
        self._pending_new_session_by_chat.pop(chat_id, None)
        stopped = await self._session_manager.stop_session(chat_id, announce=False)
        await self._send_message(
            chat_id,
            "[session stopped]" if stopped else "No active session.",
            reply_markup=self._keyboard_factory(),
        )

    async def _handle_cancel(self, chat_id: int, raw_arg: str) -> None:
        del raw_arg
        selection_cancelled = (
            self._pending_new_session_by_chat.pop(chat_id, None) is not None
        )
        interrupted = await self._session_manager.interrupt(chat_id)
        await self._send_message(
            chat_id,
            self._cancel_message(
                selection_cancelled=selection_cancelled,
                interrupted=interrupted,
            ),
            reply_markup=self._keyboard_factory(),
        )

    async def _handle_pending_provider_choice(
        self,
        chat_id: int,
        pending: PendingNewSession,
        choice: str,
    ) -> None:
        provider_name = choice.lower()
        if provider_name not in self._settings.providers:
            await self._send_message(
                chat_id,
                (
                    f"Unknown provider {choice!r}. "
                    "Choose one of the configured providers or send /cancel."
                ),
                reply_markup=self._provider_keyboard(),
            )
            return

        pending.provider_name = provider_name
        await self._send_message(
            chat_id,
            self._directory_prompt(provider_name),
            reply_markup=self._directory_keyboard(provider_name),
        )

    async def _handle_pending_directory_choice(
        self,
        chat_id: int,
        provider_name: str,
        choice: str,
    ) -> None:
        try:
            workdir = self._resolve_workdir_choice(provider_name, choice)
            await self._start_session(
                chat_id,
                provider_name,
                workdir,
                show_keyboard=False,
            )
        except ValueError as exc:
            await self._send_message(
                chat_id,
                f"Error: {exc}\n\n{self._directory_prompt(provider_name)}",
                reply_markup=self._directory_keyboard(provider_name),
            )

    async def _set_preferred_provider(self, chat_id: int, provider_name: str) -> None:
        self._ensure_provider_exists(provider_name)
        self._preferred_provider_by_chat[chat_id] = provider_name
        await self._send_message(
            chat_id,
            f"Preferred provider set to {provider_name}. "
            "Use /new to restart the session with this provider.",
            reply_markup=self._keyboard_factory(),
        )

    def _ensure_provider_exists(self, provider_name: str) -> None:
        if provider_name not in self._settings.providers:
            available = ", ".join(sorted(self._settings.providers))
            raise ValueError(
                f"Unknown provider {provider_name!r}. Available: {available}"
            )

    async def _begin_new_session(
        self,
        chat_id: int,
        provider_name: str | None = None,
    ) -> None:
        if provider_name is not None:
            self._ensure_provider_exists(provider_name)
            self._pending_new_session_by_chat[chat_id] = PendingNewSession(
                provider_name=provider_name
            )
            await self._send_message(
                chat_id,
                self._directory_prompt(provider_name),
                reply_markup=self._directory_keyboard(provider_name),
            )
            return

        preferred = self.preferred_provider(chat_id)
        self._pending_new_session_by_chat[chat_id] = PendingNewSession()
        await self._send_message(
            chat_id,
            (
                "Select provider for the new session.\n"
                f"Current preferred provider: {preferred}\n"
                "Send /cancel to abort."
            ),
            reply_markup=self._provider_keyboard(),
        )

    async def _start_session(
        self,
        chat_id: int,
        provider_name: str,
        workdir: Path,
        *,
        show_keyboard: bool = True,
    ) -> None:
        try:
            await self._session_manager.start_session(
                chat_id,
                provider_name,
                cwd=workdir,
            )
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise ValueError(
                f"Failed to start provider {provider_name}: {exc}"
            ) from exc
        self._pending_new_session_by_chat.pop(chat_id, None)
        await self._send_message(
            chat_id,
            f"[session started: {provider_name} | workdir={workdir}]",
            reply_markup=self._keyboard_factory() if show_keyboard else None,
        )

    def _parse_new_arguments(
        self,
        chat_id: int,
        raw_arg: str,
    ) -> tuple[str, str | None]:
        try:
            tokens = shlex.split(raw_arg)
        except ValueError as exc:
            raise ValueError(f"Invalid /new arguments: {exc}") from exc

        if not tokens:
            return self.preferred_provider(chat_id), None

        provider_candidate = tokens[0].lower()
        if provider_candidate in self._settings.providers:
            directory_choice = " ".join(tokens[1:]) or None
            return provider_candidate, directory_choice

        if len(tokens) > 1:
            raise ValueError("Usage: /new [provider] [directory]")

        return self.preferred_provider(chat_id), tokens[0]

    def _provider_keyboard(self) -> ReplyKeyboardMarkup:
        return self._choices_keyboard(sorted(self._settings.providers))

    def _directory_keyboard(self, provider_name: str) -> ReplyKeyboardMarkup:
        choices = directory_choices(
            self._settings.providers,
            provider_name,
            button_limit=_DIRECTORY_BUTTON_LIMIT,
        )
        return self._choices_keyboard(choices)

    def _choices_keyboard(self, choices: list[str]) -> ReplyKeyboardMarkup:
        rows: list[list[str]] = []
        for index in range(0, len(choices), _KEYBOARD_COLUMNS):
            rows.append(choices[index : index + _KEYBOARD_COLUMNS])
        rows.append(["/cancel"])
        return ReplyKeyboardMarkup(
            rows,
            resize_keyboard=True,
            one_time_keyboard=True,
        )

    def _directory_prompt(self, provider_name: str) -> str:
        return directory_prompt(
            self._settings.providers,
            provider_name,
            preview_limit=_DIRECTORY_BUTTON_LIMIT,
        )

    def _resolve_workdir_choice(self, provider_name: str, value: str) -> Path:
        return resolve_workdir_choice(self._settings.providers, provider_name, value)

    @staticmethod
    def _cancel_message(*, selection_cancelled: bool, interrupted: bool) -> str:
        if selection_cancelled and interrupted:
            return "[request cancelled]\n[new session setup cancelled]"
        if selection_cancelled:
            return "[new session setup cancelled]"
        return "[request cancelled]" if interrupted else "No active request."

    def _help_text(self, chat_id: int) -> str:
        return (
            "Commands:\n"
            "/help - show this message\n"
            "/list - list configured providers\n"
            "/use <provider> - set preferred provider for this chat\n"
            "/new [provider] [directory] - choose or start a session\n"
            "/status - show current session status\n"
            "/queue - show queued prompts\n"
            "/stop - stop the current session\n"
            "/cancel - cancel the in-flight request or /new setup\n\n"
            f"Current preferred provider: {self.preferred_provider(chat_id)}\n"
            "Use /new with no arguments to choose a provider and a direct child "
            "directory under the configured workdir.\n"
            "Plain text messages are forwarded as standalone CLI requests and "
            "queued while the provider is busy."
        )
