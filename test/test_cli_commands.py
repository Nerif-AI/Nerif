"""Tests for CLI commands (no API calls needed for check and models)."""

import subprocess
import sys


class TestCLICheck:
    def test_check_runs(self):
        """nerif check should run without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "nerif.cli.check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Nerif Configuration Check" in result.stdout
        assert "OpenAI" in result.stdout
        assert "Anthropic" in result.stdout

    def test_check_shows_providers(self):
        result = subprocess.run(
            [sys.executable, "-m", "nerif.cli.check"],
            capture_output=True,
            text=True,
        )
        assert "providers available" in result.stdout


class TestCLIModels:
    def test_models_via_main(self):
        """nerif models should list supported prefixes."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['nerif', 'models']; from nerif.cli.main import main; main()",
            ],
            capture_output=True,
            text=True,
        )
        assert "Supported Model Prefixes" in result.stdout
        assert "anthropic/" in result.stdout
        assert "gemini/" in result.stdout
        assert "ollama/" in result.stdout


class TestCLIMain:
    def test_no_args_shows_usage(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.argv = ['nerif']; from nerif.cli.main import main; exit(main())"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Commands:" in result.stdout

    def test_unknown_command(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['nerif', 'foobar']; from nerif.cli.main import main; exit(main())",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Unknown command" in result.stdout


class TestPyTyped:
    def test_py_typed_exists(self):
        """py.typed marker file should exist in the package."""
        from pathlib import Path

        import nerif

        package_dir = Path(nerif.__file__).parent
        py_typed = package_dir / "py.typed"
        assert py_typed.exists(), "py.typed marker file missing"
