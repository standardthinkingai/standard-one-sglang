# Jev benchmark sources and evaluation protocol

Sources checked on 2026-09-20. These are three different evaluations: the
TypeSafe vendor evaluation, kishida's llama.cpp model comparison, and independent
community tests. None establishes the accuracy or speed of this SGLang fork.

## TypeSafe's published evaluation

The [official workflow evaluation](https://evals.typesafe.ai/) publishes four
business workflows:

| Workflow | Decision being evaluated |
| --- | --- |
| [Security incidents](https://evals.typesafe.ai/security_incidents) | Close, queue, or contain an incident |
| [Agent trace observability](https://evals.typesafe.ai/agent_trace_observability) | Decide whether a completed agent run needs review |
| [Invoice processing](https://evals.typesafe.ai/invoice_processing) | Pay, hold, dispute, or request corrections |
| [Customer service](https://evals.typesafe.ai/customer_service) | Select the next support actions |

The published methodology uses reference probabilities averaged from GPT-6 Astra
and Claude Fable 5.1 at high thinking. The other models use their providers'
default reasoning settings. Thus its reported accuracy measures agreement with
model-derived reference decisions, not agreement with independently verified
human labels. The overview gives the four workflows equal weight.

The [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
attributes the advertised 193.6× speed and 444.6× cost comparisons to these
workflows. It says these gains are toward the high end of expected real-world
gains. Its LLM comparator uses the
[System One adapter](https://github.com/typesafe-ai/system-one-adapter-python),
which generates structured decisions and probabilities.

The site exposes methodology, workflow descriptions, and selected examples. A
downloadable complete corpus, reference-label file, workflow runner, and dataset
reuse license were not located in the inspected pages or the organization's
public repository inventory. Do not describe an independently assembled suite as
a reproduction of the official benchmark. The adapter's MIT code license does
not itself license the vendor's evaluation data.

The launch's Doom and Wikiracing segments are demonstrations. Doom uses
structured game state expressed as text, explicitly not screenshots. Wikiracing
chooses links and uses two stages when the candidate set exceeds 255. Neither is
evidence of screenshot-to-controller performance.

## The 1,191-question llama.cpp comparison

The inspected source is
[kishida/llama.cpp `docs/jev.md`](https://github.com/kishida/llama.cpp/blob/09e2b41b304f7c9a3a097859ae0db453f26217bb/docs/jev.md),
at commit `09e2b41b304f7c9a3a097859ae0db453f26217bb` on branch `jev`.

The documented protocol is:

- 1,191 evaluation questions, with two to eight choices; stated guessing accuracy
  is 0.283.
- A separate 358-question split fits a temperature by negative log-likelihood.
- Accuracy and ECE before/after temperature scaling are reported.
- Hardware is an RTX 4060 Ti with `-ngl 99 -c 16384 -np 8`.
- Latency is the mean of 50 sequential requests, each carrying one question.
- The named runner is `scripts/bench_models.py` in a separate “Jev PoC.”

That script is not present in the inspected `jev` branch tree. The exact dataset,
its composition and language distribution, split IDs, labels, source license,
and the separate PoC were not located in the author's public GitHub repositories,
gists, Hugging Face model cards, demo-space tree, or dataset inventory. This is a
search finding, not proof that the files cannot exist elsewhere. The benchmark
cannot currently be reproduced exactly from the linked repository alone.

The fork's MIT license does not establish the missing dataset's license. Do not
invent a familiar dataset name or treat this as TypeSafe's official benchmark.

## Reusable community game tests

[wondertwins/jev-benchmark](https://github.com/wondertwins/jev-benchmark/tree/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b)
is an independent, MIT-licensed evaluation, pinned here to
`1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b`. Its
[license](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/LICENSE)
must accompany copied material.

- **NPC addressee detection:** 79 author-labeled utterances across 16 categories,
  with clean, punctuation-free speech transcription, and misheard-name variants.
  It exercises Choice, Noul, and Score. Four ambiguous items are excluded from
  strict metrics; preserve and document that exclusion policy.
- **Chess:** 30 middlegame positions, 25 mate-in-one puzzles, and full-game
  experiments. Inputs range from board text to code-computed tactical facts.
  The richer inputs change the task and must be reported separately.
- Raw requests/responses, labels, runners, and reports are included. The samples
  are small and the existing results are useful as diagnostics, not a general
  model ranking.

Verified source paths:

| Purpose | Source |
| --- | --- |
| NPC fixtures | [`npcaddress/dataset.py`](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/npcaddress/dataset.py) |
| NPC question definitions | [`npcaddress/questions.py`](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/npcaddress/questions.py) |
| NPC runner | [`npcaddress/run.py`](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/npcaddress/run.py) |
| Chess runner | [`jevchess/run.py`](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/jevchess/run.py) |
| Chess position generation | [`jevchess/positions.py`](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/jevchess/positions.py) |
| Client to adapt for a local endpoint | [`jevcommon/client.py`](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/jevcommon/client.py) |

These tests consume text/structured state. They do **not** fulfill the project's
screenshot-controlled game evaluation requirement.

## Screenshot-controlled games

A visual game suite must give the model rendered frames and action descriptions.
Game rules, controller execution, and measurement remain in the harness. Do not
replace the screenshot with privileged engine coordinates, object labels,
optimal moves, or a textual description extracted from hidden state.

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) is a suitable subsequent
environment: its [game-state API](https://vizdoom.farama.org/api/python/game_state/)
provides a screen buffer, and its
[DoomGame API](https://vizdoom.farama.org/api/python/doom_game/)
provides controller actions, episode recording, and replay. Hidden engine state
may supply outcome metrics, but must not enter the policy request for a
pixels-only run. Observe the environment's and game assets' separate licenses.

Recommended records for each visual episode are scenario/version, seed, input
frame hashes, image resolution, controller action set, frame skip/action
duration, requested and actual timestamps, selected actions, distributions,
rewards, termination reason, and the complete request trace. Score actual
episode outcomes and deadline misses, not just the validity of action JSON.

Keep two timing modes separate: a paused environment measures decision quality
without time pressure; a real-time environment measures whether the controller
can act before its deadline. End-to-end latency includes frame capture, image
encoding/upload, vision preprocessing, prefill, readout, and action delivery.

## Split integrity and fair comparisons

This is the proposed protocol for new evaluations, not a description of an
unavailable upstream split:

1. Keep prompt development, calibration, and final evaluation sets disjoint.
   Fit temperature and choose thresholds only on the calibration/development
   sets, then freeze them before evaluating.
2. Split by source family. All variants/translations of an NPC utterance stay
   together. All frames of one game episode stay together; group related maps
   or scenario templates when measuring generalization. Business tickets from
   the same thread/customer/template also stay together.
3. Preserve original row IDs and record source URL, revision, license, content
   hash, split assignment, and the reason for every exclusion. Synthetic smoke
   fixtures must be labeled synthetic and kept separate from accuracy claims.
4. Give prefill-only, one-token generation, and generated-JSON baselines the same
   model revision, quantization, input evidence, option set, and reasoning
   setting. Report cold/warm prefix-cache conditions separately. Do not compare
   a prefill-only classifier only against a verbose reasoning baseline.
5. Report accuracy/macro-F1, NLL, Brier score, reliability/ECE with bin details,
   and risk versus coverage where abstention is relevant. For ordinal Score,
   include MAE against an appropriate target. Choice probabilities describe
   preference among supplied actions; they are not automatically game-win
   probabilities.
6. Record p50/p95/p99 latency, throughput at stated concurrency, question/option
   counts, text and image tokens, cache hits, backend-generated token counts,
   failures, and timeouts. Keep timed-out episodes in the outcome denominator.

## Existing SGLang implementation to compare with

[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang/tree/604664a22b2cf44c6cc499e503092ae4e3c24c03)
already uses Qwen3.6-35B-A3B and is a useful design reference. At inspected commit
`604664a22b2cf44c6cc499e503092ae4e3c24c03`, its README describes:

- A gateway over SGLang, rather than a native SGLang endpoint.
- One cache-warming request plus one request per question, each with
  `max_new_tokens=1`; sampled tokens are discarded. Backend output usage is
  correctly reported as N+1 for N questions.
- Up to 64 verified single-token answer labels, not Jev's 255 choices.
- Text-only chat state, entropy-derived confidence, and uncalibrated default
  probabilities.
- Opportunistic prefix reuse, with hybrid recurrent-state and cache-page
  limitations; unavailable cache counts are reported as unavailable.

These details prevent misleading comparisons: no autoregressive continuation
does not imply zero sampled tokens, a 64-option smoke test does not verify
255-option support, and entropy-derived confidence is not empirical accuracy.
No license file was found in the inspected repository tree, so this investigation
uses it as a reference and does not import its implementation.
