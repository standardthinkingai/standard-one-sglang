# KEV-4B public benchmark and dataset notes

Checked against the public repository on 2026-09-21. KEV's published suites are
text decision tasks, not image benchmarks. The local visual probes in this
directory are a separate pipeline diagnostic.

## Training versus evaluation

| Suite / split | Records | Questions |
| --- | ---: | ---: |
| decision-v7 training | 12,576 | 15,576 |
| decision-v7 calibration | 968 | 1,148 |
| decision-v7 development | 1,204 | 1,468 |
| decision-v7 locked test | 1,176 | 1,440 |
| transfer-v4 development | 764 | 764 |
| transfer-v4 locked test | 764 | 764 |

Sources: [decision-v7 manifest](https://github.com/jaredpalmer/kev/blob/main/evals/v7/decision-v7/manifest.json),
[transfer-v4 manifest](https://github.com/jaredpalmer/kev/blob/main/evals/v4/transfer-v4/manifest.json).
One record can contain several questions, so records and decisions are not
interchangeable units.

The training corpus contains 1,000 records from each of Banking77, BoolQ,
AG News, MNLI, SST5, Yelp, TREC, DBpedia14, Amazon Reviews, and IMDb, plus 896
policy records and 1,680 generated-rule records. The model card reports two
epochs of LoRA r16 plus a pointer head on Qwen3.5-4B-Base and says that no Jev
outputs were used for training. [Training recipe](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md#training)

The transfer suite uses six sources absent from that fine-tuning mixture:
MMLU, Emotion, TweetEval-offensive, QNLI, PAWS, and SciQ, plus held-out policy
structures. "Unseen" refers to KEV's fine-tuning split, not proof that the base
Qwen or proprietary Jev never saw these public datasets during pretraining.
[Data construction](https://github.com/jaredpalmer/kev/blob/main/kev/data.py)

## Frozen development file verified locally

Downloaded only transfer-v4 development from the public
[HF mirror at its pinned revision](https://huggingface.co/datasets/jaredpalmer/kev-suites/resolve/a957287d1c502a4e2e3b9d9d1325c2c6f27f181c/v4/transfer-v4/development.jsonl).
The file's SHA-256 matches the manifest:

```text
ff374c49c6c9f15f8a56fb274b4a4857d20497eb8dd1ac07ce01560e682a5f2e
```

Actual parsed counts: 656 `clean`, 36 `permuted`, 36 `none_present`, and 36
`none_absent` records. Clean sources contain 80 records each for the six public
datasets and `legacy_holdout`, and 96 for `composition_holdout`.

The model request contains only `state` and each question's `type`,
`instructions`, and optional `criteria`. Ground-truth `label`, source `src`,
and all `_meta` fields must be removed. The local 32-record probe takes four
clean records per source, retaining complete policy contrastive pairs. It is
not the full published benchmark and does not touch the locked test file.

## How the published benchmark scores decisions

- Headline clean accuracy uses the highest-probability answer against the
  question's label. Variant records have separate robustness measurements.
- Brier score sums squared differences from the one-hot target over choices.
- ECE uses ten equal-width bins of the maximum option probability.
- Confident errors count wrong decisions with maximum option probability at
  least 0.9. These metrics use raw maximum option probability, not either
  server's separately derived `confidence` field.
- Contrastive-pair correctness requires both sibling questions to be correct.
- Option permutations and none-of-the-above variants test answer-order
  dependence and whether the model detects an omitted correct option.

Sources: [benchmark code](https://github.com/jaredpalmer/kev/blob/main/kev/benchmark.py),
[metric implementation](https://github.com/jaredpalmer/kev/blob/main/kev/evaluate.py),
[contrastive policies](https://github.com/jaredpalmer/kev/blob/main/kev/contrastive.py).

Development selects candidates; the model card reports a final locked-test
read. The CLI's `--allow-test` option is permission to read a test suite, not
a persistent technical guarantee that nobody can evaluate it more than once.
Local model-only timing and remote HTTP timing also differ; neither should be
directly compared with this Mac's image preprocessing plus inference timing.

## Reproduce against the local API

In a separate checkout/environment of the upstream KEV repository, with this
directory's MLX server already running:

```bash
uv run python -m kev.benchmark \
  --remote http://127.0.0.1:30110 \
  --remote-model jev-latest \
  --suite evals/v4/transfer-v4 \
  --out runs/local-qwen-transfer-dev
```

Use `--remote-model jev-latest`: upstream defaults to `kev-latest`, which this
local adapter does not serve. The full development suite is larger than the
32-record local probe. [Suite loader and mirror verification](https://github.com/jaredpalmer/kev/blob/main/kev/suite.py)

Published Jev comparisons use the same items through the remote provider, but
the provider does not expose a model-weight revision, and its training corpus
is not public. Replaying the input corpus is possible; reproducing the exact
historical Jev checkpoint and its training-data exposure is not established.
[Jev client](https://github.com/jaredpalmer/kev/blob/main/kev/jev.py)
