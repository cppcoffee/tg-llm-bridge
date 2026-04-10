from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_tg_bot.providers import PreparedRequest, ProviderAdapter, ProviderSpec
from llm_tg_bot.workdirs import providers_text, resolve_workdir_choice


class _FakeAdapter(ProviderAdapter):
    def __init__(self, name: str = "fake", executable: str = "fake") -> None:
        self.name = name
        self.executable = executable

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


class WorkdirTests(unittest.TestCase):
    def test_resolve_workdir_choice_accepts_root_and_direct_child_only(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            resolved_root = root.resolve()
            child = root / "child"
            nested = child / "nested"
            child.mkdir()
            nested.mkdir()

            providers = {"fake": ProviderSpec(adapter=_FakeAdapter(), cwd=root)}

            self.assertEqual(
                resolve_workdir_choice(providers, "fake", "."),
                resolved_root,
            )
            self.assertEqual(
                resolve_workdir_choice(providers, "fake", "child"),
                child.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "direct child directory"):
                resolve_workdir_choice(providers, "fake", "child/nested")

    def test_providers_text_collapses_shared_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            providers = {
                "claude": ProviderSpec(
                    adapter=_FakeAdapter(name="claude", executable="claude"),
                    cwd=root,
                ),
                "codex": ProviderSpec(
                    adapter=_FakeAdapter(name="codex", executable="codex"),
                    cwd=root,
                ),
            }

            text = providers_text(providers)

            self.assertIn(f"Workdir root: {root}", text)
            self.assertIn("- claude: claude", text)
            self.assertIn("- codex: codex", text)
