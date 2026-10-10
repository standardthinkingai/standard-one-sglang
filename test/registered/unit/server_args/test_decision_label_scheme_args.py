"""--decision-label-scheme from the command line to what the decision worker reads.

The model worker resolves the scheme it enforces from
``get_model().decision_label_scheme`` (logits_processor) and the integrated
jev-adapter service from ``server_args.decision_label_scheme``; unit tests of
those consumers inject namespaces, so this checks the real parse and projection.
"""

import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

from sglang.srt import runtime_context as rc
from sglang.srt.arg_groups.decision_hook import handle_decision_readout_validation
from sglang.srt.decision_contract import resolve_decision_label_scheme
from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

READOUT = ["--decision-readout-path", "/models/decision.npz"]


def worker_scheme():
    """The model worker's expression (logits_processor.py) for the scheme."""
    return resolve_decision_label_scheme(
        getattr(rc.get_model(), "decision_label_scheme", None)
    )


class TestDecisionLabelSchemeArgs(CustomTestCase):
    def setUp(self):
        rc.reset_context()
        self.addCleanup(rc.reset_context)

    def publish(self, argv):
        server_args = prepare_server_args(["--model-path", "dummy", *argv])
        rc.publish(server_args, role="test")
        return server_args

    def test_flag_reaches_the_worker_and_the_adapter(self):
        for scheme in ("served", "upper"):
            with self.subTest(scheme=scheme):
                rc.reset_context()
                with mock.patch.dict(os.environ, {"JEV_LABEL_SCHEME": ""}):
                    server_args = self.publish(
                        [*READOUT, "--decision-label-scheme", scheme]
                    )
                    self.assertEqual(server_args.decision_label_scheme, scheme)
                    self.assertEqual(rc.get_model().decision_label_scheme, scheme)
                    self.assertEqual(worker_scheme(), scheme)

    def test_flag_wins_over_the_environment(self):
        with mock.patch.dict(os.environ, {"JEV_LABEL_SCHEME": "upper"}):
            self.publish([*READOUT, "--decision-label-scheme", "served"])
            self.assertEqual(worker_scheme(), "served")

    def test_unset_flag_means_environment_then_upper(self):
        self.publish(READOUT)
        self.assertIsNone(rc.get_model().decision_label_scheme)
        with mock.patch.dict(os.environ, {"JEV_LABEL_SCHEME": ""}):
            self.assertEqual(worker_scheme(), "upper")
        with mock.patch.dict(os.environ, {"JEV_LABEL_SCHEME": "served"}):
            self.assertEqual(worker_scheme(), "served")

    def test_unknown_scheme_is_a_usage_error(self):
        with redirect_stderr(StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                prepare_server_args(
                    [
                        *("--model-path", "dummy", *READOUT),
                        *("--decision-label-scheme", "lower"),
                    ]
                )
        self.assertIn("--decision-label-scheme", stderr.getvalue())

    def test_decision_settings_without_a_readout_are_refused(self):
        for argv, name in (
            (["--decision-label-scheme", "served"], "decision_label_scheme"),
            (["--decision-deadline-seconds", "30"], "decision_deadline_seconds"),
        ):
            with self.subTest(name=name):
                server_args = prepare_server_args(["--model-path", "dummy", *argv])
                with self.assertRaisesRegex(
                    ValueError, f"{name} requires --decision-readout-path"
                ):
                    handle_decision_readout_validation(server_args)
        # A plain server is unaffected.
        handle_decision_readout_validation(
            prepare_server_args(["--model-path", "dummy"])
        )


if __name__ == "__main__":
    unittest.main()
