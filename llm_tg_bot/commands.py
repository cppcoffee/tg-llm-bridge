from __future__ import annotations

from collections.abc import Awaitable, Callable

from telegram import ReplyKeyboardMarkup

from llm_tg_bot.config import Settings
from llm_tg_bot.session import SessionManager

SendMessage = Callable[[int, str, ReplyKeyboardMarkup | None], Awaitable[None]]
KeyboardFactory = Callable[[], ReplyKeyboardMarkup]

BOT_COMMANDS = frozenset(
    {
        "/help",
        "/list",
        "/use",
        "/new",
        "/status",
        "/stop",
        "/cancel",
    }
)


def command_name(text: str) -> str:
    return text.split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()


def is_bot_command(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and command_name(stripped) in BOT_COMMANDS


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

    async def handle(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        command = command_name(parts[0])
        arg = parts[1].strip().lower() if len(parts) > 1 else None

        if command == "/help":
            await self._send_message(
                chat_id,
                self._help_text(chat_id),
                reply_markup=self._keyboard_factory(),
            )
            return

        if command == "/list":
            await self._send_message(chat_id, self._providers_text())
            return

        if command == "/use":
            if not arg:
                await self._send_message(chat_id, "Usage: /use <provider>")
                return
            await self._set_preferred_provider(chat_id, arg)
            return

        if command == "/new":
            provider_name = arg or self.preferred_provider(chat_id)
            self._ensure_provider_exists(provider_name)
            try:
                await self._session_manager.start_session(chat_id, provider_name)
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                raise ValueError(
                    f"Failed to start provider {provider_name}: {exc}"
                ) from exc
            await self._send_message(
                chat_id,
                f"[session started: {provider_name}]",
                reply_markup=self._keyboard_factory(),
            )
            return

        if command == "/status":
            preferred = self.preferred_provider(chat_id)
            status = self._session_manager.status_text(chat_id)
            await self._send_message(
                chat_id, f"Preferred provider: {preferred}\n{status}"
            )
            return

        if command == "/stop":
            stopped = await self._session_manager.stop_session(chat_id, announce=False)
            await self._send_message(
                chat_id,
                "[session stopped]" if stopped else "No active session.",
            )
            return

        if command == "/cancel":
            interrupted = await self._session_manager.interrupt(chat_id)
            await self._send_message(
                chat_id,
                "[request cancelled]" if interrupted else "No active request.",
            )
            return

        await self._send_message(chat_id, "Unknown command. Use /help.")

    def preferred_provider(self, chat_id: int) -> str:
        return self._preferred_provider_by_chat.get(
            chat_id, self._settings.default_provider
        )

    async def _set_preferred_provider(self, chat_id: int, provider_name: str) -> None:
        self._ensure_provider_exists(provider_name)
        self._preferred_provider_by_chat[chat_id] = provider_name
        await self._send_message(
            chat_id,
            f"Preferred provider set to {provider_name}. "
            "Use /new to restart the session with this provider.",
        )

    def _ensure_provider_exists(self, provider_name: str) -> None:
        if provider_name not in self._settings.providers:
            available = ", ".join(sorted(self._settings.providers))
            raise ValueError(
                f"Unknown provider {provider_name!r}. Available: {available}"
            )

    def _providers_text(self) -> str:
        provider_items = sorted(self._settings.providers.items())
        workdirs = {self._format_workdir(provider.cwd) for _, provider in provider_items}

        if len(workdirs) == 1:
            shared_workdir = next(iter(workdirs))
            lines = [f"Workdir: {shared_workdir}", "Available providers:"]
            for name, provider in provider_items:
                lines.append(f"- {name}: {provider.display_command}")
            return "\n".join(lines)

        lines = ["Available providers:"]
        for name, provider in provider_items:
            workdir = self._format_workdir(provider.cwd)
            lines.append(f"- {name}: {provider.display_command} | workdir={workdir}")
        return "\n".join(lines)

    @staticmethod
    def _format_workdir(workdir: object) -> str:
        return str(workdir) if workdir else "(current working directory)"

    def _help_text(self, chat_id: int) -> str:
        return (
            "Commands:\n"
            "/help - show this message\n"
            "/list - list configured providers\n"
            "/use <provider> - set preferred provider for this chat\n"
            "/new [provider] - start or restart a provider session\n"
            "/status - show current session status\n"
            "/stop - stop the current session\n"
            "/cancel - cancel the in-flight request\n\n"
            f"Current preferred provider: {self.preferred_provider(chat_id)}\n"
            "Plain text messages are forwarded as standalone CLI requests and "
            "queued while the provider is busy."
        )
