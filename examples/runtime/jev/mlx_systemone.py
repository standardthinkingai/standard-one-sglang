"""Local, training-free Qwen3.5 vision decisions on Apple Silicon.

Uses the same pure System One scoring helpers as the SGLang endpoint, but MLX
performs an independent multimodal prefill for each question/rotation. No text
is decoded. This is an experimental local adapter, not the proprietary Jev.
"""

import argparse
import base64
import importlib.util
import io
import math
import string
import sys
import threading
import time
from pathlib import Path

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/entrypoints/systemone_protocol.py"
)
_spec = importlib.util.spec_from_file_location("jev_mlx_protocol", PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = protocol
_spec.loader.exec_module(protocol)
SystemOneRequest = protocol.SystemOneRequest


class MLXSystemOne:
    def __init__(self, model_path: str, served_model: str | None = None):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx = mx
        self.model_path = model_path
        self.served_model = served_model or model_path
        self.lock = threading.Lock()
        start = time.perf_counter()
        self.model, self.processor = load(model_path, trust_remote_code=False)
        self.model.eval()
        if self.model.config.model_type != "qwen3_5":
            raise ValueError("This local adapter is verified for Qwen3.5 only")
        if not hasattr(self.model, "vision_tower"):
            raise ValueError("A model with the vision tower is required")
        self.tokenizer = self.processor.tokenizer
        self.load_seconds = time.perf_counter() - start
        self.vision_calls = 0

    @staticmethod
    def decode_images(images):
        from PIL import Image

        decoded = []
        if len(images) > 4:
            raise ValueError("This local adapter accepts at most four images")
        for value in images:
            # Keep the local evaluation/server independent of external URLs.
            if not value.startswith("data:image/") or ";base64," not in value:
                raise ValueError("The local adapter accepts base64 image data URLs only")
            payload = value.split(",", 1)[1]
            if len(payload) > 24_000_000:
                raise ValueError("Image exceeds the local adapter size limit")
            with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as img:
                decoded.append(img.convert("RGB"))
        return decoded

    def _prefill(self, prompt, images, labels, assistant_prefix=None):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        mx = self.mx
        started = time.perf_counter()
        formatted = apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=len(images),
            enable_thinking=False,
        )
        if assistant_prefix is not None:
            formatted += assistant_prefix
        prefix_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        label_ids = []
        for label in labels:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.encode(
                formatted + label, add_special_tokens=False
            ) != prefix_ids + ids:
                raise ValueError(f"Label {label!r} is not one token at the answer boundary")
            label_ids.append(ids[0])
        if len(set(label_ids)) != len(label_ids):
            raise ValueError("Label token IDs must be distinct")
        inputs = prepare_inputs(
            self.processor,
            images=images or None,
            prompts=formatted,
            image_token_index=self.model.config.image_token_index,
        )
        if "attention_mask" in inputs:
            inputs["mask"] = inputs.pop("attention_mask")
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > 8192:
            raise ValueError("Local adapter input limit is 8192 tokens")
        image_tokens = int(
            mx.sum(inputs["input_ids"] == self.model.config.image_token_index).item()
        )
        if images and ("pixel_values" not in inputs or image_tokens == 0):
            raise RuntimeError("Image request lost its pixels or image tokens")
        preparation_ms = (time.perf_counter() - started) * 1000
        inference_start = time.perf_counter()
        lm = self.model.language_model
        lm._position_ids = None
        lm._rope_deltas = None
        # Explicit embedding preparation calls the real vision tower, without
        # vision-feature or prefix caches. Materialize to time vision separately.
        features = self.model.get_input_embeddings(**inputs)
        mx.eval(features.inputs_embeds)
        vision_ms = (time.perf_counter() - inference_start) * 1000 if images else 0.0
        if images:
            self.vision_calls += 1
        kwargs = dict(inputs)
        input_ids = kwargs.pop("input_ids")
        kwargs.update(features.to_dict())
        output = lm(
            input_ids,
            cache=lm.make_cache(),
            return_hidden=True,
            skip_logits=True,
            **kwargs,
        )
        # Project only the last prompt position; no token is appended or decoded.
        hidden = output.hidden_states[-1][:, -1:, :]
        logits = (
            lm.model.embed_tokens.as_linear(hidden)
            if lm.args.tie_word_embeddings
            else lm.lm_head(hidden)
        )[0, 0, :].astype(mx.float32)
        vocab_logprobs = logits - mx.logsumexp(logits)
        selected = vocab_logprobs[mx.array(label_ids)]
        mx.eval(selected)
        values = selected.tolist()
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Model produced nonfinite label log-probabilities")
        return values, {
            "input_tokens": input_tokens,
            "image_tokens": image_tokens,
            "images": len(images),
            "vision_forward_executed": bool(images),
            "preprocessing_ms": preparation_ms,
            "vision_embedding_ms": vision_ms,
            "inference_ms": (time.perf_counter() - inference_start) * 1000,
            "total_ms": (time.perf_counter() - started) * 1000,
            "label_token_ids": label_ids,
            "label_probability_mass": math.fsum(math.exp(v) for v in values),
        }

    def score(self, request):
        if not isinstance(request, SystemOneRequest):
            request = SystemOneRequest.model_validate(request)
        if request.model not in {self.served_model, self.model_path, "jev-latest"}:
            raise ValueError("Requested model is not loaded")
        with self.lock:
            started = time.perf_counter()
            images = self.decode_images(request.images)
            answers, details = {}, []
            for key, question in request.questions.items():
                plan = protocol.plan_question(question)
                count = len(plan.option_names)
                if count > 62:
                    raise ValueError("Local adapter supports at most 62 choices")
                labels = tuple((string.ascii_uppercase + string.ascii_lowercase + string.digits)[:count])
                orders, vectors = [], []
                for rotation in range(request.options.permutations):
                    order = protocol.rotation_order(count, rotation)
                    prompt = protocol.build_prompt(request.state, plan, labels, order)
                    vector, detail = self._prefill(
                        prompt, images, labels, request.options.assistant_prefix
                    )
                    orders.append(order)
                    vectors.append(vector)
                    details.append({"question": key, "rotation": rotation, **detail})
                answers[key] = protocol.reduce_probabilities(plan, vectors, orders, request.options)
            return {
                "model": self.served_model,
                "answers": answers,
                "usage": {
                    "input_tokens": sum(d["input_tokens"] for d in details),
                    "output_tokens": 0,
                },
                "metadata": {
                    "backend": "mlx-vlm",
                    "model_inference": True,
                    "training_performed": False,
                    "enable_thinking": False,
                    "confidence_method": "one_minus_normalized_entropy",
                    "calibration_temperature": request.options.temperature if request.options.temperature_scaling else 1.0,
                    "evaluations": len(details),
                    "vision_feature_cache": False,
                    "prefix_cache": False,
                    "wall_ms": (time.perf_counter() - started) * 1000,
                    "prefills": details,
                },
            }


def create_app(engine):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Local MLX System One vision adapter")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.served_model, "vision_enabled": True}

    @app.post("/v1/systemone")
    def score(request: SystemOneRequest):
        try:
            return engine.score(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MLX-VLM snapshot directory")
    parser.add_argument("--served-model", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--port", type=int, default=30110)
    args = parser.parse_args()
    engine = MLXSystemOne(args.model, args.served_model)
    import uvicorn

    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
