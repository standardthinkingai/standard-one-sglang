"""Render the exact served-wording prompt token IDs for each parity record.

released: the released adapter package + its native tokenizer pin
          (mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae97...), exactly as
          jev-adapter --tokenizer-model/--tokenizer-revision prepares /generate IDs.
fork:     the fork's jev_adapter service text + the engine tokenizer loaded by
          SGLang's get_tokenizer for /models/StandardOne-8B, rendered the way
          InProcessDecisionBackend._native_ids does.
"""

import asyncio
import hashlib
import json
import sys

from jev_adapter.native_tokenizer import NativeTokenizer
from jev_adapter.protocol import SystemOneRequest
from jev_adapter.service import SystemOneService

mode, data_path, out_path = sys.argv[1:4]

if mode == "released":
    native = NativeTokenizer.from_pretrained(
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "f6fae9795746f63c9be8344932f01275f3c63734",
    )
else:
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    native = NativeTokenizer(
        get_tokenizer(
            "/models/StandardOne-8B",
            tokenizer_mode="auto",
            trust_remote_code=False,
            revision="36d7999c63403acc1fd5861d5b139eee275e26d8",
        )
    )
tokenizer = native.tokenizer
print(mode, type(tokenizer).__name__, file=sys.stderr)

letters = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
labels, token_ids = [], []
for label in letters:
    ids = tokenizer.encode(label, add_special_tokens=False)
    if len(ids) == 1:
        labels.append(label)
        token_ids.append(ids[0])


class Captured(Exception):
    pass


class Capture:
    model = "standard-one-8b"

    def labels(self, count):
        return tuple(labels[:count]), tuple(token_ids[:count])

    async def evaluate(self, prompt, images, labels_, token_ids_, prefix):
        raise Captured(prompt, labels_, token_ids_)


service = SystemOneService(
    Capture(), alias="jev-latest", default_temperature=1.65, prompt_wording="served"
)


async def render(row):
    request = SystemOneRequest.model_validate({"model": "jev-latest", **row["request"]})
    try:
        await service.score(request)
    except Captured as captured:
        return captured.args
    raise RuntimeError("no evaluation captured")


with open(out_path, "w") as out:
    for line in open(data_path):
        row = json.loads(line)
        prompt, labels_, token_ids_ = asyncio.run(render(row))
        if mode == "released":
            ids = native.prepare(prompt, labels_, token_ids_, None)
        else:
            ids = native._chat([{"role": "user", "content": prompt}], continuation=False)
        out.write(
            json.dumps(
                {
                    "id": row["id"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "labels": list(labels_),
                    "label_token_ids": list(token_ids_),
                    "ids": ids,
                }
            )
            + "\n"
        )
print("done", file=sys.stderr)
