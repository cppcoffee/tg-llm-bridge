from __future__ import annotations

import os
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    command: tuple[str, ...]
    output_file: Path | None = None


class ProviderAdapter(ABC):
    name: str
    executable: str

    @abstractmethod
    def prepare_request(self, prompt: str, resume: bool) -> PreparedRequest:
        raise NotImplementedError

    @abstractmethod
    def build_response(
        self,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
        output_file: Path | None,
    ) -> str:
        raise NotImplementedError


class ClaudeAdapter(ProviderAdapter):
    name = "claude"
    executable = "claude"

    def prepare_request(self, prompt: str, resume: bool) -> PreparedRequest:
        command = [self.executable, "-p", "--output-format", "text"]
        if resume:
            command.append("--continue")
        command.append(prompt)
        return PreparedRequest(command=tuple(command))

    def build_response(
        self,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
        output_file: Path | None,
    ) -> str:
        del output_file
        return _build_text_response(stdout_text, stderr_text, return_code)


class GeminiAdapter(ProviderAdapter):
    name = "gemini"
    executable = "gemini"

    def prepare_request(self, prompt: str, resume: bool) -> PreparedRequest:
        if resume:
            command = [
                self.executable,
                "--resume",
                "latest",
                "-p",
                prompt,
                "--output-format",
                "text",
            ]
        else:
            command = [self.executable, "-p", prompt, "--output-format", "text"]
        return PreparedRequest(command=tuple(command))

    def build_response(
        self,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
        output_file: Path | None,
    ) -> str:
        del output_file
        return _build_text_response(stdout_text, stderr_text, return_code)


class CodexAdapter(ProviderAdapter):
    name = "codex"
    executable = "codex"

    def prepare_request(
        self,
        prompt: str,
        resume: bool,
        skip_git_repo_check: bool = False,
    ) -> PreparedRequest:
        fd, temp_path = tempfile.mkstemp(prefix="llm-tg-bot-codex-", suffix=".txt")
        os.close(fd)
        output_file = Path(temp_path)

        if resume:
            command = [
                self.executable,
                "exec",
                "resume",
            ]
            if skip_git_repo_check:
                command.append("--skip-git-repo-check")
            command.extend(
                [
                    "--last",
                    "--output-last-message",
                    str(output_file),
                    prompt,
                ]
            )
        else:
            command = [
                self.executable,
                "exec",
            ]
            if skip_git_repo_check:
                command.append("--skip-git-repo-check")
            command.extend(
                [
                    "--color",
                    "never",
                    "--output-last-message",
                    str(output_file),
                    prompt,
                ]
            )

        return PreparedRequest(command=tuple(command), output_file=output_file)

    def build_response(
        self,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
        output_file: Path | None,
    ) -> str:
        primary_text = _read_output_file(output_file) or _clean_output_text(stdout_text)
        return _build_response(
            primary_text,
            _add_codex_repo_check_hint(stderr_text),
            return_code,
        )


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    adapter: ProviderAdapter
    cwd: Path | None = None
    skip_git_repo_check: bool = False

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def display_command(self) -> str:
        return self.adapter.executable

    def prepare_request(self, prompt: str, resume: bool) -> PreparedRequest:
        if isinstance(self.adapter, CodexAdapter):
            return self.adapter.prepare_request(
                prompt,
                resume,
                skip_git_repo_check=self.skip_git_repo_check,
            )
        return self.adapter.prepare_request(prompt, resume)

    def build_response(
        self,
        stdout_text: str,
        stderr_text: str,
        return_code: int,
        output_file: Path | None,
    ) -> str:
        return self.adapter.build_response(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            return_code=return_code,
            output_file=output_file,
        )


_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CODEX_REPO_CHECK_ERROR = (
    "Not inside a trusted directory and --skip-git-repo-check was not specified."
)
_IGNORED_STDERR_PATTERNS = (
    "WARNING: proceeding, even though we could not update PATH",
)
_BUILTIN_ADAPTERS: tuple[ProviderAdapter, ...] = (
    CodexAdapter(),
    ClaudeAdapter(),
    GeminiAdapter(),
)


def builtin_adapters() -> tuple[ProviderAdapter, ...]:
    return _BUILTIN_ADAPTERS


def _build_text_response(stdout_text: str, stderr_text: str, return_code: int) -> str:
    return _build_response(_clean_output_text(stdout_text), stderr_text, return_code)


def _build_response(primary_text: str, stderr_text: str, return_code: int) -> str:
    parts: list[str] = []
    if primary_text:
        parts.append(primary_text)

    stderr_clean = _clean_stderr_text(stderr_text)
    if return_code != 0 and stderr_clean:
        parts.append(f"[stderr]\n{stderr_clean}")

    if return_code != 0 and not parts:
        parts.append(f"[request failed: exit code {return_code}]")

    return "\n\n".join(parts).strip()


def _read_output_file(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return _clean_output_text(path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return ""


def _clean_output_text(text: str) -> str:
    cleaned = _ANSI_ESCAPE_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in cleaned.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def _clean_stderr_text(text: str) -> str:
    cleaned = _clean_output_text(text)
    if not cleaned:
        return ""

    lines = [
        line
        for line in cleaned.splitlines()
        if not any(pattern in line for pattern in _IGNORED_STDERR_PATTERNS)
    ]
    return "\n".join(lines).strip()


def _add_codex_repo_check_hint(text: str) -> str:
    cleaned = _clean_stderr_text(text)
    if _CODEX_REPO_CHECK_ERROR not in cleaned:
        return text

    return (
        f"{cleaned}\n\n"
        "Hint: set WORKDIR to the project directory Codex should use. If you "
        "disabled the default bypass, set CODEX_SKIP_GIT_REPO_CHECK=1 to allow "
        "running outside a trusted Git worktree."
    )
