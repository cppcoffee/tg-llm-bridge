# llm-tg-bot

A Python Telegram bot that bridges chat messages to local CLI agents such as `codex`, `claude`, and `gemini`.

The bridge uses a headless request/response model instead of forwarding interactive TUI output. Telegram replies therefore look like normal CLI text instead of a partially rendered terminal screen.

## Project layout

```text
.
├── llm_tg_bot/
├── deploy/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

The project uses `pyproject.toml` as the single source of truth for packaging and dependencies. A separate `requirements.txt` is intentionally not included.

## Current behavior

- Long-polling Telegram Bot API via `python-telegram-bot`
- Separate logical sessions per chat
- Headless provider execution for `codex`, `claude`, and `gemini`
- Provider switching and session reset
- Per-chat request queueing while a provider is busy
- User allowlist with Telegram user IDs
- Idle timeout cleanup, Telegram-safe message splitting, and rate limit retry

## Quick start

1. Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

2. Copy the environment template:

```bash
cp .env.example .env
```

3. Fill in `.env`:

- `TELEGRAM_BOT_TOKEN`: Telegram bot token
- `TELEGRAM_ALLOWED_USER_IDS`: Recommended configuration. Comma-separated Telegram numeric user IDs
- Setting the allowlist variable to `*` disables validation and should only be used during development
- `DEFAULT_PROVIDER`: Preferred default provider, for example `codex`
- `WORKDIR`: Working directory shared by all providers. Recommended for `codex`. If omitted, the bridge uses the directory where the bot process started
- `CODEX_SKIP_GIT_REPO_CHECK`: Optional. Defaults to enabled for `codex`. Set it to `0` only if you want to require a trusted Git worktree
- `SESSION_IDLE_TIMEOUT_SECONDS`: Optional. Defaults to `2700` seconds, which closes an idle session after 45 minutes

4. Ensure at least one supported CLI is available in `PATH`:

- `codex`
- `claude`
- `gemini`

5. Start the bot:

```bash
python -m llm_tg_bot.main
```

After installation, you can also run the generated console script:

```bash
llm-tg-bot
```

## Telegram commands

- `/help`: Show help
- `/list`: Show configured providers
- `/use <provider>`: Set the preferred provider for the current chat
- `/new [provider]`: Start a fresh logical session, optionally with a different provider
- `/status`: Show the current chat session status, including queued requests
- `/stop`: Stop and forget the current session
- `/cancel`: Cancel the in-flight provider request and continue with any queued requests

Plain text messages are sent as standalone headless CLI requests and the resulting text is sent back to Telegram. If a chat sends more messages while a request is still running, the bridge queues them and replies with how many requests are ahead. Slash-prefixed text that is not a bot command is also forwarded directly.

## Codex workdir notes

`codex` refuses to run outside a trusted Git worktree unless `--skip-git-repo-check` is passed.

- For normal use, set `WORKDIR` to the repository root you want Codex to operate on
- The bridge enables `--skip-git-repo-check` for `codex` by default, so no extra configuration is required for non-repository directories
- Set `CODEX_SKIP_GIT_REPO_CHECK=0` only if you explicitly want Codex to require a trusted Git worktree
- `/list` shows the effective workdir used by each provider
