"""Jev-shaped decisions using the existing prefill-only generation path.

The model supplies label log-probabilities; Python constructs the response.
No text or JSON is sampled. The existing chat adapter owns image handling and
chat templates, so multimodal scoring uses the same preprocessing as chat.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import string
import time
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING

from sglang.srt.decision_contract import resolve_decision_label_scheme

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from sglang.srt.entrypoints.systemone_protocol import (
    SystemOneRequest,
    build_prompt,
    plan_question,
    reduce_probabilities,
    rotation_order,
)

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class SystemOneRoute(APIRoute):
    """Keep 422 validation local to this API, independent of chat's 400 handler."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request):
            service = getattr(request.app.state, "systemone_service", None)
            if service is not None:
                from sglang.srt.entrypoints.systemone import handle_systemone

                return await handle_systemone(request, service)
            try:
                return await handler(request)
            except RequestValidationError as exc:
                error = exc.errors()[0]
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "invalid_request",
                            "message": error["msg"],
                            "field": ".".join(
                                str(part) for part in error["loc"] if part != "body"
                            ),
                        }
                    },
                )

        return validated


router = APIRouter(route_class=SystemOneRoute)


class SystemOneError(ValueError):
    def __init__(self, code: str, message: str, field: str = "", status: int = 422):
        super().__init__(message)
        self.code, self.field, self.status = code, field, status


def _label_candidates(scheme: str):
    """Candidate label order; the same rule as jev_adapter.labels.

    "upper": A..Z, then AA..ZZ, then AAA..ZZZ (uppercase only, by length then
    alphabetically). "served": the original order, A..Z, a..z, 0..9, 000..999,
    AA..ZZ, aa..zz. Both start with A..Z, so at most 26 options never differ.
    """
    if scheme == "upper":
        for length in (1, 2, 3):
            yield from (
                "".join(chars)
                for chars in itertools.product(string.ascii_uppercase, repeat=length)
            )
        return
    if scheme != "served":
        raise ValueError(f"unknown label scheme {scheme!r}")
    # Human-readable labels need not be a single character. Short letter
    # combinations extend the original fork's 62-label ceiling to 255 choices.
    yield from string.ascii_uppercase + string.ascii_lowercase + string.digits
    yield from (f"{i:03d}" for i in range(1000))
    yield from (
        "".join(chars) for chars in itertools.product(string.ascii_uppercase, repeat=2)
    )
    yield from (
        "".join(chars) for chars in itertools.product(string.ascii_lowercase, repeat=2)
    )


def select_label_tokens(tokenizer, count: int, scheme: str):
    """Find distinct, round-trippable single-token labels, excluding specials."""
    labels, ids, seen = [], [], set()
    special = set(getattr(tokenizer, "all_special_ids", ()))
    for label in _label_candidates(scheme):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] in seen or encoded[0] in special:
            continue
        if tokenizer.decode(encoded) != label:
            continue
        labels.append(label)
        ids.append(encoded[0])
        seen.add(encoded[0])
        if len(labels) == count:
            return tuple(labels), tuple(ids)
    raise SystemOneError(
        "insufficient_label_tokens",
        f"Tokenizer supports only {len(labels)} verified labels; {count} required.",
        "questions",
    )


class SystemOneServing:
    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        chat_serving,
        *,
        chat_request_factory=None,
    ):
        self.manager = tokenizer_manager
        self.chat = chat_serving
        self.chat_request_factory = chat_request_factory
        max_running_requests = getattr(
            getattr(tokenizer_manager, "server_args", None),
            "max_running_requests",
            None,
        )
        self.max_concurrency = max_running_requests or 32
        self._capacity = asyncio.Semaphore(self.max_concurrency)
        # Labels beyond Z follow JEV_LABEL_SCHEME (default upper), as in the
        # integrated decision service.
        self.label_scheme = resolve_decision_label_scheme(None)
        self._labels = lru_cache(maxsize=255)(
            lambda count: select_label_tokens(
                self.manager.tokenizer, count, self.label_scheme
            )
        )

    def _validate_backend(self, request):
        if not self.manager.is_generation or self.manager.tokenizer is None:
            raise SystemOneError(
                "unsupported_model", "A generation model with a tokenizer is required."
            )
        names = {self.manager.served_model_name, self.manager.model_path, "jev-latest"}
        if request.model not in names:
            raise SystemOneError(
                "model_not_found",
                "Requested model is not served by this process.",
                "model",
                404,
            )
        if self.manager.config_value("speculative_algorithm") is not None:
            raise SystemOneError(
                "speculation_not_supported",
                "Disable speculative decoding and MTP for prefill-only System One scoring.",
            )
        if request.images and not self.manager.model_config.is_multimodal:
            raise SystemOneError(
                "images_not_supported",
                "The loaded model does not support images.",
                "images",
            )

    def _prepare(self, request, plan, order, labels, token_ids, raw_request):
        factory = self.chat_request_factory
        if factory is None:
            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

            factory = ChatCompletionRequest
        content = []
        for image in request.images or []:
            content.append({"type": "image_url", "image_url": {"url": image}})
        content.append(
            {"type": "text", "text": build_prompt(request.state, plan, labels, order)}
        )
        chat_request = factory(
            model=self.manager.served_model_name,
            messages=[{"role": "user", "content": content}],
            chat_template_kwargs={"enable_thinking": False},
            stream=False,
            max_tokens=0,
        )
        adapted, _ = self.chat._convert_to_internal_request(chat_request, raw_request)
        tokenizer = self.manager.tokenizer
        prefix = request.options.assistant_prefix
        if prefix:
            if adapted.text is not None:
                adapted.text += prefix
            else:
                # Retokenize the boundary rather than blindly joining token IDs.
                text = tokenizer.decode(adapted.input_ids)
                if (
                    tokenizer.encode(text, add_special_tokens=False)
                    != adapted.input_ids
                ):
                    raise SystemOneError(
                        "unsupported_assistant_prefix",
                        "This tokenizer cannot round-trip prompt IDs; an assistant prefix would alter the prompt.",
                        "options.assistant_prefix",
                    )
                text += prefix
                adapted.input_ids = tokenizer.encode(text, add_special_tokens=False)

        # Ensure the scored token means the written label at the actual answer
        # boundary. Leading-space merges otherwise silently score wrong tokens.
        tail = (
            adapted.text[-128:]
            if adapted.text is not None
            else tokenizer.decode(adapted.input_ids[-64:])
        )
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        if adapted.text is None and tail_ids != adapted.input_ids[-64:]:
            raise SystemOneError(
                "unsupported_tokenizer_boundary",
                "This tokenizer cannot round-trip the answer boundary for label validation.",
            )
        for label, token_id in zip(labels, token_ids):
            if tokenizer.encode(tail + label, add_special_tokens=False) != tail_ids + [
                token_id
            ]:
                raise SystemOneError(
                    "invalid_label_boundary",
                    "Label tokenization changes at the assistant prefix; use a prefix ending in a newline.",
                    "options.assistant_prefix",
                )
        # Replace inherited sampling defaults, including penalties and grammar.
        # The zero-token path computes vocabulary log-probs without sampling.
        adapted.sampling_params = {
            "max_new_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        adapted.return_logprob = True
        adapted.logprob_start_len = -1
        adapted.token_ids_logprob = list(token_ids)
        adapted.top_logprobs_num = 0
        adapted.return_text_in_logprobs = False
        adapted.stream = False
        adapted.require_reasoning = False
        adapted.rid = "systemone-" + uuid.uuid4().hex
        return adapted

    async def _evaluate(self, adapted, token_ids, raw_request):
        async with self._capacity:
            iterator = self.manager.generate_request(adapted, raw_request)
            finished = False
            try:
                result = await iterator.__anext__()
                finished = True
            finally:
                if not finished:
                    self.manager.abort_request(adapted.rid)
                await iterator.aclose()
        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(
                    "Unexpected batched result for one System One question."
                )
            result = result[0]
        meta = result["meta_info"]
        if meta.get("completion_tokens") != 0:
            raise RuntimeError(
                "System One backend generated tokens despite max_new_tokens=0."
            )
        positions = meta.get("output_token_ids_logprobs")
        if not positions:
            raise RuntimeError(
                "Backend did not return prefill label log-probabilities."
            )
        values = {entry[1]: entry[0] for entry in positions[0]}
        if any(
            token_id not in values or values[token_id] is None for token_id in token_ids
        ):
            raise RuntimeError("Backend omitted requested label log-probabilities.")
        if any(
            isinstance(values[token_id], bool)
            or not isinstance(values[token_id], (int, float))
            or not math.isfinite(values[token_id])
            for token_id in token_ids
        ):
            raise RuntimeError("Backend returned nonfinite label log-probabilities.")
        prompt_tokens = meta.get("prompt_tokens")
        cached_tokens = meta.get("cached_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise RuntimeError("Backend returned invalid prompt-token usage.")
        if cached_tokens is not None and (
            type(cached_tokens) is not int or not 0 <= cached_tokens <= prompt_tokens
        ):
            raise RuntimeError("Backend returned invalid cached-token usage.")
        return (
            [values[token_id] for token_id in token_ids],
            prompt_tokens,
            cached_tokens,
        )

    async def score(self, request: SystemOneRequest, raw_request=None):
        self._validate_backend(request)
        started = time.perf_counter()
        prepared, groups = [], []
        for question_id, question in request.questions.items():
            plan = plan_question(question)
            n = len(plan.option_names)
            labels, token_ids = self._labels(n)
            orders = [rotation_order(n, i) for i in range(request.options.permutations)]
            start = len(prepared)
            for order in orders:
                adapted = self._prepare(
                    request, plan, order, labels, token_ids, raw_request
                )
                prepared.append((adapted, token_ids))
            groups.append((question_id, plan, orders, start, len(prepared)))

        tasks = [
            asyncio.create_task(self._evaluate(adapted, ids, raw_request))
            for adapted, ids in prepared
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        answers = {}
        for question_id, plan, orders, start, end in groups:
            answers[question_id] = reduce_probabilities(
                plan, [row[0] for row in results[start:end]], orders, request.options
            )
        cached_tokens = [row[2] for row in results]
        return {
            "model": self.manager.served_model_name,
            "answers": answers,
            "usage": {
                "input_tokens": sum(row[1] for row in results),
                "output_tokens": 0,
            },
            "metadata": {
                "confidence_method": "one_minus_normalized_entropy",
                "calibration_temperature": request.options.temperature
                if request.options.temperature_scaling
                else 1.0,
                "evaluations": len(prepared),
                "service_max_concurrency": self.max_concurrency,
                "label_scheme": self.label_scheme,
                "adapter_elapsed_ms": (time.perf_counter() - started) * 1000,
                **(
                    {"cached_tokens": sum(cached_tokens)}
                    if all(value is not None for value in cached_tokens)
                    else {}
                ),
            },
        }


@router.post("/v1/systemone")
async def systemone(request: SystemOneRequest, raw_request: Request):
    try:
        return await raw_request.app.state.systemone_serving.score(request, raw_request)
    except SystemOneError as exc:
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {"code": exc.code, "message": str(exc), "field": exc.field}
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": str(exc)}},
        )
    except RuntimeError:
        logger.exception("System One scoring failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "scoring_failed",
                    "message": "Backend could not produce valid prefill scores.",
                }
            },
        )
