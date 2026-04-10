from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from llm_tg_bot.providers import ProviderSpec
from llm_tg_bot.rendering import OutgoingMessage, RenderMode

logger = logging.getLogger(__name__)

ProcessTracker = Callable[[asyncio.subprocess.Process | None], None]


@dataclass(frozen=True, slots=True)
class RequestExecutionResult:
    completed_at: float
    message: OutgoingMessage | None
    succeeded: bool


async def run_provider_request(
    provider: ProviderSpec,
    prompt: str,
    *,
    resume: bool,
    process_tracker: ProcessTracker | None = None,
) -> RequestExecutionResult:
    output_file = None
    process: asyncio.subprocess.Process | None = None
    try:
        request = provider.prepare_request(prompt, resume=resume)
        output_file = request.output_file
        logger.info("Running provider=%s command=%s", provider.name, request.command)
        process = await asyncio.create_subprocess_exec(
            *request.command,
            cwd=str(provider.cwd) if provider.cwd else None,
            env=_child_environment(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process_tracker is not None:
            process_tracker(process)

        stdout_bytes, stderr_bytes = await process.communicate()
        response = provider.build_response(
            stdout_text=stdout_bytes.decode("utf-8", errors="replace"),
            stderr_text=stderr_bytes.decode("utf-8", errors="replace"),
            return_code=process.returncode,
            output_file=output_file,
        )
        return RequestExecutionResult(
            completed_at=time.monotonic(),
            message=_response_message(response, process.returncode),
            succeeded=process.returncode == 0,
        )
    except asyncio.CancelledError:
        if process and process.returncode is None:
            await terminate_process(process)
        raise
    finally:
        if output_file:
            with contextlib.suppress(FileNotFoundError):
                output_file.unlink()
        if process_tracker is not None:
            process_tracker(None)


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _response_message(response: str, return_code: int) -> OutgoingMessage | None:
    if response:
        render_mode = RenderMode.MARKDOWN if return_code == 0 else RenderMode.PLAIN
        return OutgoingMessage(response, render_mode=render_mode)
    if return_code != 0:
        return OutgoingMessage(f"[request failed: exit code {return_code}]\n")
    return None


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    term = env.get("TERM", "").strip().lower()
    if not term or term == "dumb":
        env["TERM"] = "xterm-256color"
    env.setdefault("COLORTERM", "truecolor")
    return env
