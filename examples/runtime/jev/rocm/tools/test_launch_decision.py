"""Check profile arguments without Docker or a GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCH = Path(__file__).resolve().parents[1] / "launch-decision.sh"


class LaunchProfileTests(unittest.TestCase):
    def launch(self, profile, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / "docker"
            docker.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            docker.chmod(0o755)
            env = {**os.environ, "PATH": directory + ":" + os.environ["PATH"],
                   "GPU_PROFILE": profile, "MEM_FRACTION": "0.15",
                   "AITER_OPS": "0", "MAX_RUNNING_REQUESTS": "32", "KVFREE": "1",
                   "ATTENTION_BACKEND": "", "IMAGE": "", **overrides}
            return subprocess.run(["bash", str(LAUNCH)], env=env, text=True, capture_output=True)

    def test_amd_baseline_and_opt_in(self):
        for profile in ("gfx950", "gfx942"):
            for enabled in ("0", "1"):
                with self.subTest(profile=profile, enabled=enabled):
                    result = self.launch(profile, AITER_OPS=enabled, MAX_RUNNING_REQUESTS="64")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = result.stdout.splitlines()
                    self.assertIn("SGLANG_USE_AITER=" + enabled, args)
                    self.assertEqual(args[args.index("--attention-backend") + 1], "triton")
                    self.assertEqual(args[args.index("--max-running-requests") + 1], "64")
                    self.assertNotIn("--prefill-only-disable-kv-cache", args)

    def test_nvidia_kvfree_and_paged(self):
        for profile, backend in (("sm90", "fa3"), ("sm100", "fa4")):
            for kvfree in ("0", "1"):
                with self.subTest(profile=profile, kvfree=kvfree):
                    result = self.launch(profile, KVFREE=kvfree)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = result.stdout.splitlines()
                    self.assertEqual(args[args.index("--attention-backend") + 1], backend)
                    self.assertEqual("--prefill-only-disable-kv-cache" in args, kvfree == "1")
                    self.assertNotIn("SGLANG_USE_AITER=0", args)

    def test_invalid_controls_fail_before_docker(self):
        for overrides in ({"AITER_OPS": "2"}, {"MAX_RUNNING_REQUESTS": "0"}, {"MAX_RUNNING_REQUESTS": "-1"}):
            with self.subTest(overrides=overrides):
                result = self.launch("gfx942", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        result = self.launch("sm90", AITER_OPS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
