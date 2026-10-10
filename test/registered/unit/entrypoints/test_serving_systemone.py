import asyncio
import json
import math
import string
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from sglang.srt.entrypoints.serving_systemone import (
    SystemOneError,
    SystemOneServing,
    router,
    select_label_tokens,
)
from sglang.srt.entrypoints.systemone_protocol import SystemOneRequest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class CharacterTokenizer:
    all_special_ids = ()

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)


class FakeChatAdapter:
    def __init__(self):
        self.requests = []

    def _convert_to_internal_request(self, request, raw_request):
        self.requests.append(request)
        content = request.messages[0]["content"]
        text = content[-1]["text"] + "\nAssistant:\n"
        return SimpleNamespace(
            text=None,
            input_ids=CharacterTokenizer().encode(text),
            image_data=[part["image_url"]["url"] for part in content[:-1]],
            sampling_params={"max_new_tokens": 99},
        ), request


class FakeManager:
    is_generation = True
    served_model_name = "test-qwen"
    model_path = "test-qwen"
    tokenizer = CharacterTokenizer()
    model_config = SimpleNamespace(is_multimodal=True)

    def __init__(self, *, failure=None, delay=0, cached_tokens=None):
        self.requests, self.aborted = [], []
        self.failure, self.delay = failure, delay
        self.cached_tokens = cached_tokens
        self.speculation = None
        self.active = 0
        self.peak = 0

    def config_value(self, key):
        return self.speculation

    async def generate_request(self, request, raw_request):
        request_index = len(self.requests)
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.failure == "runtime":
                raise RuntimeError("no scores")
            entries = [
                [math.log(0.8 if index == 0 else 0.1), token, None]
                for index, token in enumerate(request.token_ids_logprob)
            ]
            if self.failure == "missing":
                entries = entries[:-1]
            if self.failure == "nonfinite":
                entries[0][0] = float("nan")
            meta_info = {
                "prompt_tokens": 20,
                "completion_tokens": 1 if self.failure == "decode" else 0,
                "output_token_ids_logprobs": [entries],
            }
            if self.cached_tokens is not None:
                meta_info["cached_tokens"] = self.cached_tokens[request_index]
            yield {"meta_info": meta_info}
        finally:
            self.active -= 1

    def abort_request(self, rid):
        self.aborted.append(rid)


def payload():
    return {
        "model": "test-qwen",
        "state": {"text": "화면과 문의를 판단"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "담당 부서",
                "criteria": {"billing": "청구", "technical": "오류"},
            },
            "yes": {"type": "noul", "instructions": "환불 요청인가?"},
            "urgency": {
                "type": "score",
                "instructions": "긴급도",
                "criteria": ["낮음", "중간", "높음"],
            },
        },
    }


class TestSystemOneServing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager, self.chat = FakeManager(), FakeChatAdapter()
        self.service = SystemOneServing(
            self.manager, self.chat, chat_request_factory=SimpleNamespace
        )

    async def test_mixed_questions_and_image_use_zero_generation_adapter(self):
        body = payload()
        body["images"] = ["data:image/png;base64,aGVsbG8="]
        result = await self.service.score(SystemOneRequest.model_validate(body))
        self.assertEqual(result["answers"]["route"]["choice"], "billing")
        self.assertAlmostEqual(result["answers"]["yes"]["noul"], 8 / 9)
        self.assertAlmostEqual(result["answers"]["urgency"]["score"], 0.3)
        self.assertEqual(result["usage"], {"input_tokens": 60, "output_tokens": 0})
        self.assertNotIn("cached_tokens", result["metadata"])
        self.assertTrue(math.isfinite(result["metadata"]["adapter_elapsed_ms"]))
        self.assertGreaterEqual(result["metadata"]["adapter_elapsed_ms"], 0)
        self.assertEqual(len(self.manager.requests), 3)
        for adapted, chat in zip(self.manager.requests, self.chat.requests):
            self.assertEqual(adapted.sampling_params["max_new_tokens"], 0)
            self.assertEqual(adapted.token_ids_logprob[0], ord("A"))
            self.assertEqual(adapted.image_data, body["images"])
            self.assertEqual(adapted.logprob_start_len, -1)
            self.assertTrue(adapted.return_logprob)
            self.assertFalse(adapted.require_reasoning)
            self.assertEqual(chat.chat_template_kwargs, {"enable_thinking": False})
        prompts = [
            CharacterTokenizer().decode(item.input_ids)
            for item in self.manager.requests
        ]
        self.assertNotIn("담당 부서", prompts[1])
        self.assertNotIn("환불 요청인가?", prompts[2])

    async def test_cached_tokens_sum_actual_engine_metadata(self):
        self.manager.cached_tokens = [1, 2, 3]
        result = await self.service.score(SystemOneRequest.model_validate(payload()))

        self.assertEqual(result["metadata"]["cached_tokens"], 6)
        self.assertEqual(result["usage"]["input_tokens"], 60)

    async def test_capacity_uses_resolved_max_running_requests(self):
        self.manager.server_args = SimpleNamespace(max_running_requests=2)
        self.manager.delay = 0.01
        service = SystemOneServing(
            self.manager, self.chat, chat_request_factory=SimpleNamespace
        )
        body = payload()
        body["questions"] = {"route": body["questions"]["route"]}
        body["options"] = {"permutations": 4}

        result = await service.score(SystemOneRequest.model_validate(body))

        self.assertEqual(len(self.manager.requests), 4)
        self.assertEqual(self.manager.peak, 2)
        self.assertEqual(result["metadata"]["service_max_concurrency"], 2)

    async def test_permutations_are_unrotated_before_averaging(self):
        body = payload()
        body["questions"] = {"route": body["questions"]["route"]}
        body["options"] = {"permutations": 2, "return_logprobs": True}
        result = await self.service.score(SystemOneRequest.model_validate(body))
        self.assertAlmostEqual(
            result["answers"]["route"]["probabilities"]["billing"], 0.5
        )
        self.assertAlmostEqual(result["answers"]["route"]["confidence"], 0)
        self.assertEqual(len(result["answers"]["route"]["logprobs"]), 2)

    async def test_model_speculation_and_image_validation_precede_inference(self):
        for field in ("model", "speculation", "images"):
            with self.subTest(field=field):
                body = payload()
                self.manager.speculation = None
                self.manager.model_config = SimpleNamespace(is_multimodal=True)
                if field == "model":
                    body["model"] = "unknown"
                elif field == "speculation":
                    self.manager.speculation = "NEXTN"
                else:
                    self.manager.model_config.is_multimodal = False
                    body["images"] = ["https://example.com/screen.png"]
                with self.assertRaises(SystemOneError):
                    await self.service.score(SystemOneRequest.model_validate(body))
        self.assertEqual(self.manager.requests, [])

    async def test_requested_rotation_count_is_preserved_when_greater_than_options(
        self,
    ):
        body = payload()
        body["questions"] = {"route": body["questions"]["route"]}
        body["options"] = {"permutations": 3}
        result = await self.service.score(SystemOneRequest.model_validate(body))
        self.assertEqual(result["metadata"]["evaluations"], 3)
        self.assertEqual(len(self.manager.requests), 3)

    async def test_never_fabricates_missing_scores_or_zero_token_usage(self):
        for failure in ("missing", "decode", "nonfinite"):
            with self.subTest(failure=failure):
                self.manager.failure = failure
                with self.assertRaises(RuntimeError):
                    await self.service.score(SystemOneRequest.model_validate(payload()))

    async def test_cancelled_http_work_aborts_pending_engine_requests(self):
        self.manager.delay = 10
        task = asyncio.create_task(
            self.service.score(SystemOneRequest.model_validate(payload()))
        )
        for _ in range(100):
            if len(self.manager.requests) == 3:
                break
            await asyncio.sleep(0.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.manager.aborted), 3)
        self.assertEqual(self.manager.active, 0)

    def test_255_labels_fail_explicitly_for_a_character_only_tokenizer(self):
        with self.assertRaisesRegex(SystemOneError, "only 62"):
            select_label_tokens(CharacterTokenizer(), 255, "served")
        # Upper labels never fall back to lowercase or digits.
        with self.assertRaisesRegex(SystemOneError, "only 26"):
            select_label_tokens(CharacterTokenizer(), 255, "upper")

    def test_upper_labels_continue_with_two_letter_single_tokens(self):
        tokenizer = PairTokenizer({"AA", "AB", "AD", "ZZ", "ab"})
        labels, ids = select_label_tokens(tokenizer, 30, "upper")
        self.assertEqual(
            labels, tuple(string.ascii_uppercase) + ("AA", "AB", "AD", "ZZ")
        )
        self.assertEqual(len(set(ids)), 30)
        served, _ = select_label_tokens(tokenizer, 30, "served")
        self.assertEqual(served[26:], ("a", "b", "c", "d"))
        for count in range(2, 27):
            with self.subTest(count=count):
                self.assertEqual(
                    select_label_tokens(tokenizer, count, "upper"),
                    select_label_tokens(tokenizer, count, "served"),
                )
                self.assertEqual(
                    select_label_tokens(tokenizer, count, "upper")[0],
                    tuple(string.ascii_uppercase[:count]),
                )

    async def test_label_scheme_follows_the_environment_and_is_reported(self):
        with patch.dict("os.environ", {"JEV_LABEL_SCHEME": ""}):
            default = SystemOneServing(
                self.manager, self.chat, chat_request_factory=SimpleNamespace
            )
        self.assertEqual(default.label_scheme, "upper")
        with patch.dict("os.environ", {"JEV_LABEL_SCHEME": "served"}):
            served = SystemOneServing(
                self.manager, self.chat, chat_request_factory=SimpleNamespace
            )
        self.assertEqual(served.label_scheme, "served")
        result = await served.score(SystemOneRequest.model_validate(payload()))
        self.assertEqual(result["metadata"]["label_scheme"], "served")
        with patch.dict("os.environ", {"JEV_LABEL_SCHEME": "lower"}):
            with self.assertRaisesRegex(ValueError, "label scheme"):
                SystemOneServing(
                    self.manager, self.chat, chat_request_factory=SimpleNamespace
                )


# Raw encode/decode results of the pinned Ministral-3 tokenizer
# (Ministral-3-8B-Instruct-2512-BF16 @ f6fae97) for every one- and two-letter
# uppercase candidate, recorded from tekken.json and cross-checked against the
# release tokenizer.json; byte-identical to jev-adapter's
# tests/data/ministral3-label-probe.json. expected_upper is the 255-label list
# the trainer and the jev-adapter use.
PROBE = json.loads(
    (Path(__file__).parent / "data" / "ministral3-label-probe.json").read_text()
)


class ProbeTokenizer:
    """Replays the recorded tokenizer; a string outside the table is an error."""

    all_special_ids = tuple(
        token
        for first, last in PROBE["special_id_ranges"]
        for token in range(first, last + 1)
    )

    def encode(self, text, add_special_tokens=False):
        return list(PROBE["encode"][text])

    def decode(self, ids):
        return "".join(PROBE["decode"][str(token)] for token in ids)


class TestRecordedMinistral3Labels(unittest.TestCase):
    def test_upper_labels_match_the_shared_list(self):
        expected = PROBE["expected_upper"]
        self.assertEqual(PROBE["schema"], "jev-tokenizer-probe-v1")
        for count in (2, 26, 27, 77, 151, 255):
            with self.subTest(count=count):
                # Three-letter candidates are absent from the table, so
                # reaching one would raise.
                labels, ids = select_label_tokens(ProbeTokenizer(), count, "upper")
                self.assertEqual(labels, tuple(expected["labels"][:count]))
                self.assertEqual(ids, tuple(expected["token_ids"][:count]))
        labels, _ = select_label_tokens(ProbeTokenizer(), 255, "upper")
        self.assertEqual(
            (labels[26], labels[76], labels[150], labels[254]),
            ("AA", "CG", "FT", "MH"),
        )


class PairTokenizer(CharacterTokenizer):
    """Characters are tokens; listed two-letter strings are single tokens too."""

    def __init__(self, pairs):
        self.pairs = {pair: 70000 + index for index, pair in enumerate(sorted(pairs))}
        self.names = {token: pair for pair, token in self.pairs.items()}

    def encode(self, text, add_special_tokens=False):
        if text in self.pairs:
            return [self.pairs[text]]
        return super().encode(text)

    def decode(self, ids):
        return "".join(self.names.get(token) or chr(token) for token in ids)


class TestSystemOneHTTP(unittest.TestCase):
    def setUp(self):
        self.manager = FakeManager()
        app = FastAPI()
        # The real SGLang application changes ordinary validation errors to
        # 400; this API's route must still retain its local 422 contract.
        app.add_exception_handler(
            RequestValidationError,
            lambda request, exc: JSONResponse(
                status_code=400, content={"error": "chat validation"}
            ),
        )
        app.state.systemone_serving = SystemOneServing(
            self.manager, FakeChatAdapter(), chat_request_factory=SimpleNamespace
        )
        app.include_router(router)
        self.client = TestClient(app)

    def test_native_route_schema_and_error_status(self):
        response = self.client.post("/v1/systemone", json=payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["answers"]), {"route", "yes", "urgency"})
        self.assertEqual(
            self.client.post("/v1/systemone", json={"model": "test-qwen"}).status_code,
            422,
        )
        body = payload()
        body["model"] = "wrong"
        self.assertEqual(self.client.post("/v1/systemone", json=body).status_code, 404)

    def test_backend_failure_is_not_a_fake_probability_distribution(self):
        for failure in ("runtime", "nonfinite"):
            with self.subTest(failure=failure):
                self.manager.failure = failure
                response = self.client.post("/v1/systemone", json=payload())
                self.assertEqual(response.status_code, 500)
                self.assertEqual(response.json()["error"]["code"], "scoring_failed")

    def test_integrated_service_uses_same_route_without_legacy_scoring(self):
        service = object()
        self.client.app.state.systemone_service = service
        handler = AsyncMock(return_value=JSONResponse({"integrated": True}))
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.entrypoints.systemone": SimpleNamespace(
                    handle_systemone=handler
                )
            },
        ):
            response = self.client.post("/v1/systemone", json=payload())
        self.assertEqual(response.json(), {"integrated": True})
        self.assertIs(handler.await_args.args[1], service)
        self.assertEqual(self.manager.requests, [])
        routes = [route for route in router.routes if route.path == "/v1/systemone"]
        self.assertEqual(len(routes), 1)

    def test_disabled_integrated_service_preserves_legacy_endpoint(self):
        self.client.app.state.systemone_service = None
        response = self.client.post("/v1/systemone", json=payload())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.manager.requests)


if __name__ == "__main__":
    unittest.main()
