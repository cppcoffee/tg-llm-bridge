from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tg_llm_bridge.providers import ProviderSpec

logger = logging.getLogger(__name__)

OutputHandler = Callable[[int, str], Awaitable[None]]


@dataclass(slots=True)
class SessionRecord:
    chat_id: int
    provider: ProviderSpec
    last_activity: float = field(default_factory=time.monotonic)
    request_count: int = 0
    active_task: asyncio.Task[None] | None = None
    active_process: asyncio.subprocess.Process | None = None

    @property
    def is_busy(self) -> bool:
        return self.active_task is not None and not self.active_task.done()


class SessionManager:
    def __init__(
        self,
        providers: dict[str, ProviderSpec],
        idle_timeout_seconds: int,
        output_callback: OutputHandler,
    ) -> None:
        self._providers = providers
        self._idle_timeout_seconds = idle_timeout_seconds
        self._output_callback = output_callback
        self._records: dict[int, SessionRecord] = {}

    async def start_session(self, chat_id: int, provider_name: str) -> SessionRecord:
        await self.stop_session(chat_id, announce=False)
        provider = self._providers[provider_name]
        record = SessionRecord(chat_id=chat_id, provider=provider)
        self._records[chat_id] = record
        return record

    async def get_or_start_session(
        self, chat_id: int, provider_name: str
    ) -> SessionRecord:
        record = self._records.get(chat_id)
        if record and record.provider.name == provider_name:
            return record
        return await self.start_session(chat_id, provider_name)

    async def send_text(
        self, chat_id: int, text: str, provider_name: str
    ) -> SessionRecord:
        record = await self.get_or_start_session(chat_id, provider_name)
        if record.is_busy:
            raise RuntimeError(
                "Provider is busy. Wait for the current response or use /cancel."
            )

        record.last_activity = time.monotonic()
        record.active_task = asyncio.create_task(self._run_request(record, text))
        return record

    def has_session(self, chat_id: int) -> bool:
        return chat_id in self._records

    def active_provider_name(self, chat_id: int) -> str | None:
        record = self._records.get(chat_id)
        if not record:
            return None
        return record.provider.name

    async def interrupt(self, chat_id: int) -> bool:
        record = self._records.get(chat_id)
        if not record or not record.active_process:
            return False

        await _terminate_process(record.active_process)
        if record.active_task:
            record.active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await record.active_task
        return True

    async def stop_session(self, chat_id: int, announce: bool = True) -> bool:
        record = self._records.pop(chat_id, None)
        if not record:
            return False

        if record.active_process:
            await _terminate_process(record.active_process)
        if record.active_task:
            record.active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await record.active_task

        if announce:
            await self._output_callback(chat_id, "[session stopped]\n")
        return True

    def status_text(self, chat_id: int) -> str:
        record = self._records.get(chat_id)
        if not record:
            return "No active session."

        idle_seconds = int(time.monotonic() - record.last_activity)
        return (
            f"Active session: {record.provider.name}\n"
            f"Command: {record.provider.display_command}\n"
            f"Mode: headless request/response\n"
            f"Requests: {record.request_count}\n"
            f"Busy: {'yes' if record.is_busy else 'no'}\n"
            f"Idle: {idle_seconds}s"
        )

    async def stop_idle_sessions(self) -> None:
        if self._idle_timeout_seconds <= 0:
            return

        now = time.monotonic()
        stale_chat_ids = [
            chat_id
            for chat_id, record in self._records.items()
            if not record.is_busy
            and now - record.last_activity >= self._idle_timeout_seconds
        ]
        for chat_id in stale_chat_ids:
            await self.stop_session(chat_id, announce=False)
            await self._output_callback(
                chat_id, "[session closed due to idle timeout]\n"
            )

    async def _run_request(self, record: SessionRecord, prompt: str) -> None:
        output_file = None
        process: asyncio.subprocess.Process | None = None
        try:
            request = record.provider.prepare_request(
                prompt,
                resume=record.request_count > 0,
            )
            output_file = request.output_file
            logger.info(
                "Running provider=%s command=%s", record.provider.name, request.command
            )
            process = await asyncio.create_subprocess_exec(
                *request.command,
                cwd=str(record.provider.cwd) if record.provider.cwd else None,
                env=_child_environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            record.active_process = process
            stdout_bytes, stderr_bytes = await process.communicate()
            record.last_activity = time.monotonic()

            response = record.provider.build_response(
                stdout_text=stdout_bytes.decode("utf-8", errors="replace"),
                stderr_text=stderr_bytes.decode("utf-8", errors="replace"),
                return_code=process.returncode,
                output_file=output_file,
            )
            if process.returncode == 0:
                record.request_count += 1

            if response:
                await self._output_callback(record.chat_id, response)
            elif process.returncode != 0:
                await self._output_callback(
                    record.chat_id,
                    f"[request failed: exit code {process.returncode}]\n",
                )
        except asyncio.CancelledError:
            if process and process.returncode is None:
                await _terminate_process(process)
            raise
        except Exception as exc:
            logger.exception("Provider request failed for chat_id=%s", record.chat_id)
            await self._output_callback(record.chat_id, f"[request failed: {exc}]\n")
        finally:
            if output_file:
                with contextlib.suppress(FileNotFoundError):
                    output_file.unlink()
            record.active_process = None
            record.active_task = None


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    term = env.get("TERM", "").strip().lower()
    if not term or term == "dumb":
        env["TERM"] = "xterm-256color"
    env.setdefault("COLORTERM", "truecolor")
    return env
