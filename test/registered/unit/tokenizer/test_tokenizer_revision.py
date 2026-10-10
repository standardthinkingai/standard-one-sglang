"""Pinned tokenizer revisions must reach every Hugging Face loading path."""

import sys
import types
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

from sglang.srt.utils.hf_transformers import tokenizer as tokenizer_module

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REVISION = "a" * 40


class TestTokenizerRevision(unittest.TestCase):
    def setUp(self):
        for target, kwargs in (
            ("check_gguf_file", {"return_value": False}),
            ("_resolve_tokenizer_name", {"side_effect": lambda name, args: name}),
            ("is_bare_tekken_checkpoint", {"return_value": False}),
            ("_apply_post_load_fixes", {"side_effect": lambda tok, name, rev: tok}),
        ):
            patched = patch.object(tokenizer_module, target, **kwargs)
            setattr(self, target, patched.start())
            self.addCleanup(patched.stop)

    def test_explicit_revision_reaches_auto_loader_and_postload(self):
        loaded = object()
        with patch.object(
            tokenizer_module.AutoTokenizer, "from_pretrained", return_value=loaded
        ) as hf_loader:
            result = tokenizer_module.get_tokenizer(
                "org/model", tokenizer_revision=REVISION
            )

        self.assertIs(result, loaded)
        self.assertEqual(hf_loader.call_args.kwargs["revision"], REVISION)
        self.assertNotIn("tokenizer_revision", hf_loader.call_args.kwargs)
        self.assertEqual(self.is_bare_tekken_checkpoint.call_args.args[1], REVISION)
        self.assertEqual(self._apply_post_load_fixes.call_args.args[2], REVISION)

    def test_caller_revision_reaches_direct_class_retry(self):
        generic = type("TokenizersBackend", (), {})()
        loaded = object()
        with (
            patch.object(
                tokenizer_module.AutoTokenizer,
                "from_pretrained",
                return_value=generic,
            ) as hf_loader,
            patch.object(
                tokenizer_module,
                "_load_tokenizer_by_declared_class",
                return_value=loaded,
            ) as direct_loader,
        ):
            result = tokenizer_module.get_tokenizer("org/model", revision=REVISION)

        self.assertIs(result, loaded)
        self.assertEqual(hf_loader.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["revision"] == REVISION for call in hf_loader.call_args_list
            )
        )
        self.assertEqual(direct_loader.call_args.kwargs["revision"], REVISION)
        self.assertEqual(self._apply_post_load_fixes.call_args.args[2], REVISION)

    def test_bare_tekken_loader_uses_same_revision(self):
        self.is_bare_tekken_checkpoint.return_value = True
        loaded = object()
        mistral_module = types.ModuleType("transformers.tokenization_mistral_common")
        fake_class = type(
            "MistralCommonTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda *args, **kwargs: loaded)},
        )
        with patch.object(fake_class, "from_pretrained", return_value=loaded) as load:
            mistral_module.MistralCommonTokenizer = fake_class
            with patch.dict(
                sys.modules,
                {"transformers.tokenization_mistral_common": mistral_module},
            ):
                result = tokenizer_module.get_tokenizer(
                    "org/model", tokenizer_revision=REVISION
                )

        self.assertIs(result, loaded)
        load.assert_called_once_with("org/model", revision=REVISION)
        self.assertEqual(self.is_bare_tekken_checkpoint.call_args.args[1], REVISION)
        self.assertEqual(self._apply_post_load_fixes.call_args.args[2], REVISION)

    def test_duplicate_revision_must_agree(self):
        with patch.object(
            tokenizer_module.AutoTokenizer, "from_pretrained", return_value=object()
        ) as hf_loader:
            tokenizer_module.get_tokenizer(
                "org/model", tokenizer_revision=REVISION, revision=REVISION
            )
            self.assertEqual(hf_loader.call_args.kwargs["revision"], REVISION)
            self.assertNotIn("tokenizer_revision", hf_loader.call_args.kwargs)
            with self.assertRaisesRegex(ValueError, "must match"):
                tokenizer_module.get_tokenizer(
                    "org/model", tokenizer_revision=REVISION, revision="other"
                )
            self.assertEqual(hf_loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
