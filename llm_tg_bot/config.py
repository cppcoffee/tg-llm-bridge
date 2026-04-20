from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from llm_tg_bot.providers import ProviderSpec, builtin_adapters


@dataclass(frozen=True, slots=True)
class Settings:
    bot_tokens: list[str]
    allow_all_users: bool
    allowed_user_ids: frozenset[int]
    default_provider: str
    poll_timeout_seconds: int
    telegram_connection_pool_size: int
    telegram_pool_timeout_seconds: float
    message_max_chars: int
    session_idle_timeout_seconds: int
    log_level: str
    providers: dict[str, ProviderSpec]


def load_settings() -> Settings:
    load_dotenv()

    bot_tokens = _load_bot_tokens()
    providers = _load_providers()
    default_provider = os.getenv("DEFAULT_PROVIDER", "codex").strip().lower()
    if default_provider not in providers:
        raise ValueError(
            f"DEFAULT_PROVIDER={default_provider!r} is not configured. "
            f"Available providers: {', '.join(sorted(providers))}"
        )

    allow_all_users, allowed_user_ids = _load_allowed_users(
        os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
    )
    telegram_connection_pool_size = _int_env("TELEGRAM_CONNECTION_POOL_SIZE", 8)
    if telegram_connection_pool_size <= 0:
        raise ValueError("TELEGRAM_CONNECTION_POOL_SIZE must be greater than 0")
    telegram_pool_timeout_seconds = _float_env("TELEGRAM_POOL_TIMEOUT_SECONDS", 5.0)
    if telegram_pool_timeout_seconds <= 0:
        raise ValueError("TELEGRAM_POOL_TIMEOUT_SECONDS must be greater than 0")

    return Settings(
        bot_tokens=bot_tokens,
        allow_all_users=allow_all_users,
        allowed_user_ids=allowed_user_ids,
        default_provider=default_provider,
        poll_timeout_seconds=_int_env("POLL_TIMEOUT_SECONDS", 30),
        telegram_connection_pool_size=telegram_connection_pool_size,
        telegram_pool_timeout_seconds=telegram_pool_timeout_seconds,
        message_max_chars=_int_env("MESSAGE_MAX_CHARS", 4000),
        session_idle_timeout_seconds=_int_env("SESSION_IDLE_TIMEOUT_SECONDS", 2700),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        providers=providers,
    )


def _load_bot_tokens() -> list[str]:
    raw = _require_env("TELEGRAM_BOT_TOKENS")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ValueError("TELEGRAM_BOT_TOKENS must contain at least one token")
    return tokens


def _load_providers() -> dict[str, ProviderSpec]:
    workdir = _optional_path_env("WORKDIR") or Path.cwd()
    codex_skip_git_repo_check = _bool_env("CODEX_SKIP_GIT_REPO_CHECK", default=True)
    providers: dict[str, ProviderSpec] = {}
    for adapter in builtin_adapters():
        if not _command_exists(adapter.executable):
            continue
        providers[adapter.name] = ProviderSpec(
            adapter=adapter,
            cwd=workdir,
            skip_git_repo_check=(
                codex_skip_git_repo_check if adapter.name == "codex" else False
            ),
        )

    if not providers:
        raise ValueError(
            "No providers available. Install codex, claude, or gemini "
            "and ensure the executables are available in PATH."
        )

    return providers


def _load_allowed_users(raw_user_ids: str) -> tuple[bool, frozenset[int]]:
    if raw_user_ids == "*":
        return True, frozenset()

    allowed_user_ids = _parse_allowed_user_ids(raw_user_ids)
    if not allowed_user_ids:
        raise ValueError(
            "Configure TELEGRAM_ALLOWED_USER_IDS. "
            "Use '*' only for development."
        )

    return False, allowed_user_ids


def _parse_allowed_user_ids(raw_value: str) -> frozenset[int]:
    if not raw_value:
        return frozenset()

    user_ids: set[int] = set()
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        try:
            user_id = int(value)
        except ValueError as exc:
            raise ValueError(
                f"TELEGRAM_ALLOWED_USER_IDS contains a non-integer value: {value!r}"
            ) from exc
        if user_id <= 0:
            raise ValueError(
                f"TELEGRAM_ALLOWED_USER_IDS must contain positive integers: {value!r}"
            )
        user_ids.add(user_id)

    return frozenset(user_ids)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{name} is not a directory: {path}")
    return path


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: 1, 0, true, false, yes, no, on, off"
    )


def _command_exists(executable: str) -> bool:
    if "/" in executable:
        candidate = Path(executable).expanduser()
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(executable) is not None
