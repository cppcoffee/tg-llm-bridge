from __future__ import annotations

from pathlib import Path

from llm_tg_bot.providers import ProviderSpec


def format_workdir(workdir: Path | None) -> str:
    return str(workdir) if workdir else "(current working directory)"


def providers_text(providers: dict[str, ProviderSpec]) -> str:
    provider_items = sorted(providers.items())
    workdirs = {format_workdir(provider.cwd) for _, provider in provider_items}

    if len(workdirs) == 1:
        shared_workdir = next(iter(workdirs))
        lines = [
            f"Workdir root: {shared_workdir}",
            "Available providers:",
        ]
        for name, provider in provider_items:
            lines.append(f"- {name}: {provider.display_command}")
        lines.append("")
        lines.append("Use /new to choose a provider and a direct child directory.")
        return "\n".join(lines)

    lines = ["Available providers:"]
    for name, provider in provider_items:
        lines.append(
            f"- {name}: {provider.display_command} | workdir={format_workdir(provider.cwd)}"
        )
    lines.append("")
    lines.append("Use /new to choose a provider and a direct child directory.")
    return "\n".join(lines)


def directory_choices(
    providers: dict[str, ProviderSpec],
    provider_name: str,
    *,
    button_limit: int,
) -> list[str]:
    return [".", *visible_child_directory_names(providers, provider_name)[:button_limit]]


def directory_prompt(
    providers: dict[str, ProviderSpec],
    provider_name: str,
    *,
    preview_limit: int,
) -> str:
    root = session_root(providers, provider_name)
    visible_directories = visible_child_directory_names(providers, provider_name)
    lines = [
        f"Select workdir for {provider_name} under {root}",
        "Use . for the root directory.",
    ]

    if visible_directories:
        preview = visible_directories[:preview_limit]
        lines.append("Direct child directories:")
        lines.extend(f"- {name}" for name in preview)
        if len(visible_directories) > len(preview):
            remaining = len(visible_directories) - len(preview)
            lines.append(
                f"- ... ({remaining} more; you can type a direct child directory name manually)"
            )
    else:
        lines.append(
            "No visible child directories were found. "
            "You can still type a direct child directory name manually."
        )

    lines.append("")
    lines.append("Send /cancel to abort.")
    return "\n".join(lines)


def resolve_workdir_choice(
    providers: dict[str, ProviderSpec],
    provider_name: str,
    value: str,
) -> Path:
    choice = value.strip()
    if not choice:
        raise ValueError("Directory selection cannot be empty.")

    root = session_root(providers, provider_name)
    if choice == ".":
        return root

    raw_path = Path(choice).expanduser()
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Directory does not exist: {choice}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to resolve directory {choice!r}: {exc}") from exc

    if not resolved.is_dir():
        raise ValueError(f"Not a directory: {choice}")
    if resolved == root:
        return root
    if resolved.parent != root:
        raise ValueError(
            "Choose the root directory (.) or a direct child directory "
            "of the configured workdir."
        )
    return resolved


def visible_child_directory_names(
    providers: dict[str, ProviderSpec],
    provider_name: str,
) -> list[str]:
    root = session_root(providers, provider_name)
    try:
        directories = [
            child.name
            for child in root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]
    except OSError as exc:
        raise ValueError(f"Failed to inspect workdir {root}: {exc}") from exc
    return sorted(directories, key=str.lower)


def session_root(providers: dict[str, ProviderSpec], provider_name: str) -> Path:
    provider = _provider_spec(providers, provider_name)
    root = provider.cwd or Path.cwd()
    try:
        resolved = root.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Configured workdir does not exist: {root}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to resolve workdir {root}: {exc}") from exc

    if not resolved.is_dir():
        raise ValueError(f"Configured workdir is not a directory: {resolved}")
    return resolved


def _provider_spec(
    providers: dict[str, ProviderSpec],
    provider_name: str,
) -> ProviderSpec:
    try:
        return providers[provider_name]
    except KeyError as exc:
        available = ", ".join(sorted(providers))
        raise ValueError(
            f"Unknown provider {provider_name!r}. Available: {available}"
        ) from exc
