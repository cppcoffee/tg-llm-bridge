from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from llm_tg_bot.providers import PreparedRequest, ProviderAdapter, ProviderSpec
from llm_tg_bot.rendering import OutgoingMessage, RenderMode
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


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int | None = 0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class SessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_provider_output_uses_markdown_rendering(self) -> None:
        outputs: list[tuple[int, OutgoingMessage]] = []

        async def output_callback(chat_id: int, message: OutgoingMessage) -> None:
            outputs.append((chat_id, message))

        manager = SessionManager(
            providers={"fake": ProviderSpec(adapter=_FakeAdapter())},
            idle_timeout_seconds=5,
            output_callback=output_callback,
        )
        record = await manager.start_session(9, "fake")
        process = _FakeProcess(stdout=b"**hello**", returncode=0)

        with patch(
            "llm_tg_bot.request_runner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ):
            await manager._run_request(record, "prompt")

        self.assertEqual(
            outputs,
            [(9, OutgoingMessage("**hello**", render_mode=RenderMode.MARKDOWN))],
        )

    async def test_failed_provider_output_stays_plain_text(self) -> None:
        outputs: list[tuple[int, OutgoingMessage]] = []

        async def output_callback(chat_id: int, message: OutgoingMessage) -> None:
            outputs.append((chat_id, message))

        class _FailingAdapter(_FakeAdapter):
            def build_response(
                self,
                stdout_text: str,
                stderr_text: str,
                return_code: int,
                output_file: Path | None,
            ) -> str:
                del stdout_text, output_file
                return f"[stderr]\n{stderr_text}\ncode={return_code}"

        manager = SessionManager(
            providers={"fake": ProviderSpec(adapter=_FailingAdapter())},
            idle_timeout_seconds=5,
            output_callback=output_callback,
        )
        record = await manager.start_session(9, "fake")
        process = _FakeProcess(stderr=b"boom", returncode=1)

        with patch(
            "llm_tg_bot.request_runner.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ):
            await manager._run_request(record, "prompt")

        self.assertEqual(
            outputs,
            [(9, OutgoingMessage("[stderr]\nboom\ncode=1"))],
        )

    async def test_stop_idle_sessions_announces_closure(self) -> None:
        outputs: list[tuple[int, OutgoingMessage]] = []

        async def output_callback(chat_id: int, message: OutgoingMessage) -> None:
            outputs.append((chat_id, message))

        manager = SessionManager(
            providers={"fake": ProviderSpec(adapter=_FakeAdapter())},
            idle_timeout_seconds=5,
            output_callback=output_callback,
        )
        record = await manager.start_session(9, "fake")
        record.last_activity = time.monotonic() - 10

        await manager.stop_idle_sessions()

        self.assertFalse(manager.has_session(9))
        self.assertEqual(
            outputs,
            [(9, OutgoingMessage("[session closed due to idle timeout]\n"))],
        )
