"""Optional real Qwen tokenizer tests; no weights or GPU are used."""

import json
import os
import unittest
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from sglang.srt.entrypoints.serving_systemone import select_label_tokens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@unittest.skipUnless(
    os.environ.get("JEV_TOKENIZER_DIR"),
    "Set JEV_TOKENIZER_DIR to downloaded Qwen tokenizer files",
)
class TestActualQwenTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(os.environ["JEV_TOKENIZER_DIR"])
        cls.backend = Tokenizer.from_file(str(cls.directory / "tokenizer.json"))
        cls.config = json.loads((cls.directory / "tokenizer_config.json").read_text())

        class Adapter:
            all_special_ids = ()

            def encode(self, text, add_special_tokens=False):
                return cls.backend.encode(
                    text, add_special_tokens=add_special_tokens
                ).ids

            def decode(self, ids):
                return cls.backend.decode(ids, skip_special_tokens=False)

        cls.tokenizer = Adapter()

    def test_255_distinct_labels_round_trip_at_non_thinking_answer_boundary(self):
        for scheme in ("upper", "served"):
            with self.subTest(scheme=scheme):
                self._check_boundary(*select_label_tokens(self.tokenizer, 255, scheme))

    def _check_boundary(self, labels, ids):
        self.assertEqual(len(set(ids)), 255)
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        )
        template = environment.from_string(self.config["chat_template"])
        for with_image in (False, True):
            content = [
                {"type": "text", "text": "Choose the best option.\nA: yes\nB: no"}
            ]
            if with_image:
                content.insert(0, {"type": "image", "image": "screen.png"})
            prompt = template.render(
                messages=[{"role": "user", "content": content}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self.assertIn("</think>", prompt)
            prefix = self.tokenizer.encode(prompt)
            for label, token_id in zip(labels, ids):
                with self.subTest(image=with_image, label=label):
                    self.assertEqual(
                        self.tokenizer.encode(prompt + label), prefix + [token_id]
                    )


if __name__ == "__main__":
    unittest.main()
