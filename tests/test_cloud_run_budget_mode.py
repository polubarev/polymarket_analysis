from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ENTRYPOINT_PATH = ROOT_DIR / "scripts" / "cloud_run_entrypoint.sh"
DEPLOY_SCRIPT_PATH = ROOT_DIR / "scripts" / "deploy_cloud_run_jobs.sh"


class CloudRunEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workdir = Path(self.temp_dir.name)
        self.bin_dir = self.workdir / "bin"
        self.bin_dir.mkdir()
        self.capture_path = self.workdir / "pipeline_args.txt"
        self._write_executable(
            self.bin_dir / "polymarket-pipeline",
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ENTRYPOINT_CAPTURE\"\n",
        )

    def _write_executable(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _run_entrypoint(self, extra_env: dict[str, str]) -> list[str]:
        env = os.environ.copy()
        env.update(
            {
                "ENV_FILE": str(self.workdir / "missing.env"),
                "ENTRYPOINT_CAPTURE": str(self.capture_path),
                "PATH": f"{self.bin_dir}:{env['PATH']}",
            }
        )
        env.update(extra_env)
        subprocess.run(
            ["bash", str(ENTRYPOINT_PATH)],
            cwd=str(ROOT_DIR),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return self.capture_path.read_text(encoding="utf-8").splitlines()

    def test_entrypoint_supports_analyze_command(self) -> None:
        args = self._run_entrypoint(
            {
                "PIPELINE_COMMAND": "analyze",
                "OUTPUT_DIR": "/tmp/polymarket-data",
                "RUN_SIGNALS": "true",
            }
        )

        self.assertEqual(args[0], "analyze")
        self.assertIn("--output-dir", args)
        self.assertIn("/tmp/polymarket-data", args)
        self.assertIn("--run-signals", args)

    def test_entrypoint_defaults_to_run_command(self) -> None:
        args = self._run_entrypoint({"OUTPUT_DIR": "/tmp/polymarket-data"})

        self.assertEqual(args[0], "run")


class CloudRunDeployConfigTests(unittest.TestCase):
    def test_budget_mode_crons_and_job_envs_are_present(self) -> None:
        script = DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn('HOURLY_CRON="${HOURLY_CRON:-10 3 * * *}"', script)
        self.assertIn('NIGHTLY_CRON="${NIGHTLY_CRON:-25 3 * * 6}"', script)
        self.assertIn('RESEARCH_CRON="${RESEARCH_CRON:-30 5 * * 6}"', script)

        self.assertIn("PIPELINE_PROFILE=ingest-daily", script)
        self.assertIn("PIPELINE_COMMAND=run", script)
        self.assertIn("MAX_EVENTS=1000", script)
        self.assertIn("INGEST_VOLUME=true", script)
        self.assertIn("SNAPSHOT_ORDERBOOK=false", script)
        self.assertIn("INCLUDE_RESOLVED=false", script)
        self.assertIn("RUN_SIGNALS=false", script)
        self.assertIn("RUN_BACKTEST=false", script)
        self.assertIn("GENERATE_CANDIDATES=false", script)

        self.assertIn("PIPELINE_PROFILE=reconcile-weekly", script)
        self.assertIn("NO_INCREMENTAL_PRICES=true", script)
        self.assertIn("INCLUDE_RESOLVED=true", script)

        self.assertIn("PIPELINE_PROFILE=research-weekly", script)
        self.assertIn("PIPELINE_COMMAND=analyze", script)
        self.assertIn("INGEST_VOLUME=false", script)
        self.assertIn("SIGNAL_DEBUG=true", script)


if __name__ == "__main__":
    unittest.main()
