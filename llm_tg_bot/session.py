from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from llm_tg_bot.providers import ProviderSpec
from llm_tg_bot.rendering import OutgoingMessage
from llm_tg_bot.request_runner import run_provider_request, terminate_process
from llm_tg_bot.workdirs import format_workdir

logger = logging.getLogger(__name__)

OutputHandler = Callable[[int, OutgoingMessage], Awaitable[None]]
RequestStartedHandler = Callable[[int, asyncio.Task[None]], None]


@dataclass(slots=True)
class SessionRecord:
    chat_id: int
    provider: ProviderSpec
    last_activity: float = field(default_factory=time.monotonic)
    request_count: int = 0
    active_task: asyncio.Task[None] | None = None
    active_process: asyncio.subprocess.Process | None = None
    pending_prompts: deque[str] = field(default_factory=deque)

    @property
    def is_busy(self) -> bool:
        return self.active_task is not None and not self.active_task.done()

    @property
    def queued_count(self) -> int:
        return len(self.pending_prompts)


@dataclass(frozen=True, slots=True)
class SendResult:
    record: SessionRecord
    queued_ahead: int


class SessionManager:
    def __init__(
        self,
        providers: dict[str, ProviderSpec],
        idle_timeout_seconds: int,
        output_callback: OutputHandler,
        request_started_callback: RequestStartedHandler | None = None,
    ) -> None:
        self._providers = providers
        self._idle_timeout_seconds = idle_timeout_seconds
        self._output_callback = output_callback
        self._request_started_callback = request_started_callback
        self._records: dict[int, SessionRecord] = {}

    async def start_session(
        self,
        chat_id: int,
        provider_name: str,
        *,
        cwd: Path | None = None,
    ) -> SessionRecord:
        await self.stop_session(chat_id, announce=False)
        provider = self._provider_for_session(provider_name, cwd)
        record = SessionRecord(chat_id=chat_id, provider=provider)
        self._records[chat_id] = record
        return record

    async def get_or_start_session(
        self,
        chat_id: int,
        provider_name: str,
    ) -> SessionRecord:
        record = self._records.get(chat_id)
        if record and record.provider.name == provider_name:
            return record
        return await self.start_session(chat_id, provider_name)

    async def send_text(
        self,
        chat_id: int,
        text: str,
        provider_name: str,
    ) -> SendResult:
        record = await self.get_or_start_session(chat_id, provider_name)
        self._sweep_completed_task(record)
        queued_ahead = record.queued_count + (1 if record.is_busy else 0)
        record.pending_prompts.append(text)
        record.last_activity = time.monotonic()
        self._ensure_active_request(record)
        return SendResult(record=record, queued_ahead=queued_ahead)

    def has_session(self, chat_id: int) -> bool:
        return chat_id in self._records

    def active_provider_name(self, chat_id: int) -> str | None:
        record = self._records.get(chat_id)
        if record is None:
            return None
        return record.provider.name

    async def interrupt(self, chat_id: int) -> bool:
        record = self._records.get(chat_id)
        if record is None:
            return False
        return await self._cancel_active_request(record)

    async def stop_session(self, chat_id: int, announce: bool = True) -> bool:
        record = self._records.pop(chat_id, None)
        if record is None:
            return False

        await self._cancel_active_request(record)
        record.pending_prompts.clear()

        if announce:
            await self._output_callback(chat_id, OutgoingMessage("[session stopped]\n"))
        return True

    def status_text(self, chat_id: int) -> str:
        record = self._records.get(chat_id)
        if record is None:
            return "No active session."

        self._ensure_active_request(record)
        idle_seconds = int(time.monotonic() - record.last_activity)
        return (
            f"Active session: {record.provider.name}\n"
            f"Command: {record.provider.display_command}\n"
            f"Workdir: {format_workdir(record.provider.cwd)}\n"
            f"Mode: headless request/response\n"
            f"Requests: {record.request_count}\n"
            f"Busy: {'yes' if record.is_busy else 'no'}\n"
            f"Queued: {record.queued_count}\n"
            f"Idle: {idle_seconds}s"
        )

    def queue_text(self, chat_id: int) -> str:
        record = self._records.get(chat_id)
        if record is None:
            return "No active session."

        if not record.pending_prompts:
            return "Queue is empty."

        queue_list = list(record.pending_prompts)
        lines = [f"Queue ({len(queue_list)} item(s)):"]
        for index, prompt in enumerate(queue_list, 1):
            # Truncate long prompts for display
            display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            lines.append(f"{index}. {display_prompt}")

        return "\n".join(lines)

    async def stop_idle_sessions(self) -> None:
        if self._idle_timeout_seconds <= 0:
            return

        now = time.monotonic()
        stale_chat_ids: list[int] = []
        for chat_id, record in self._records.items():
            self._ensure_active_request(record)
            if (
                not record.is_busy
                and now - record.last_activity >= self._idle_timeout_seconds
            ):
                stale_chat_ids.append(chat_id)

        for chat_id in stale_chat_ids:
            await self.stop_session(chat_id, announce=False)
            await self._output_callback(
                chat_id,
                OutgoingMessage("[session closed due to idle timeout]\n"),
            )

    def _ensure_active_request(self, record: SessionRecord) -> bool:
        self._sweep_completed_task(record)
        if record.active_task is not None or not record.pending_prompts:
            return False

        prompt = record.pending_prompts.popleft()
        task = asyncio.create_task(self._run_request(record, prompt))
        record.active_task = task
        task.add_done_callback(
            lambda completed_task: self._on_request_done(record, completed_task)
        )
        if self._request_started_callback is not None:
            self._request_started_callback(record.chat_id, task)
        return True

    def _on_request_done(
        self,
        record: SessionRecord,
        completed_task: asyncio.Task[None],
    ) -> None:
        if record.active_task is completed_task:
            record.active_task = None

        if self._records.get(record.chat_id) is not record:
            return

        self._ensure_active_request(record)

    @staticmethod
    def _sweep_completed_task(record: SessionRecord) -> None:
        if record.active_task is not None and record.active_task.done():
            record.active_task = None

    async def _cancel_active_request(self, record: SessionRecord) -> bool:
        self._sweep_completed_task(record)
        had_active_request = (
            record.active_task is not None or record.active_process is not None
        )

        if record.active_process is not None:
            await terminate_process(record.active_process)

        if record.active_task is not None:
            record.active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await record.active_task
            record.active_task = None

        record.active_process = None
        return had_active_request

    def _provider_for_session(
        self,
        provider_name: str,
        cwd: Path | None,
    ) -> ProviderSpec:
        provider = self._providers[provider_name]
        if cwd is None or cwd == provider.cwd:
            return provider
        return ProviderSpec(
            adapter=provider.adapter,
            cwd=cwd,
            skip_git_repo_check=provider.skip_git_repo_check,
        )

    def _track_active_process(
        self,
        record: SessionRecord,
        process: asyncio.subprocess.Process | None,
    ) -> None:
        record.active_process = process

    async def _run_request(self, record: SessionRecord, prompt: str) -> None:
        try:
            result = await run_provider_request(
                record.provider,
                prompt,
                resume=record.request_count > 0,
                process_tracker=lambda process: self._track_active_process(
                    record, process
                ),
            )
            record.last_activity = result.completed_at
            if result.succeeded:
                record.request_count += 1
            if result.message is not None:
                await self._output_callback(record.chat_id, result.message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Provider request failed for chat_id=%s", record.chat_id)
            await self._output_callback(
                record.chat_id,
                OutgoingMessage(f"[request failed: {exc}]\n"),
            )
        finally:
            record.active_process = None
