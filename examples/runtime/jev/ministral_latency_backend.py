"""Measure training-free Ministral 3 decision latency with MLX-VLM.

This backend evaluates a prompt once and reads the logits for A/B/C/D at the
answer boundary. It does not generate a token or calculate an accuracy score.
Use the same prompt with ``images=[]`` and ``images=[pil_image]`` to measure
the incremental image cost on the same loaded multimodal model.

Disk reads, image-file decoding, model loading, HTTP transport, and queueing
are outside ``measure``. CPU image preprocessing and GPU image encoding are
inside it. The caller should warm up every input shape, exclude warmups from
steady-state statistics, and serialize requests to avoid GPU contention.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path


class MinistralLatencyBackend:
    """A local ``mistral3`` VLM with a ``ministral3`` language decoder."""

    labels = ("A", "B", "C", "D")

    def __init__(self, model_path: str | Path):
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError("Pass an already-downloaded local model directory")
        self.mx = mx
        self.model_path = str(model_path)
        self._apply_chat_template = apply_chat_template
        self._prepare_inputs = prepare_inputs
        self._lock = threading.Lock()
        mx.synchronize()
        started = time.perf_counter()
        self.model, self.processor = load(
            self.model_path, trust_remote_code=False, lazy=False
        )
        self.model.eval()
        if (
            self.model.config.model_type != "mistral3"
            or self.model.config.text_config.model_type != "ministral3"
            or not hasattr(self.model, "vision_tower")
        ):
            raise ValueError(
                "Expected a Mistral 3 multimodal model with Ministral 3 decoder"
            )
        self.tokenizer = self.processor.tokenizer
        self.language_model = self.model.language_model
        self.image_token_id = self.model.config.image_token_index
        if self.image_token_id is None:
            raise ValueError("The model configuration has no image token ID")
        self.label_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)
            for label in self.labels
        ]
        if any(len(ids) != 1 for ids in self.label_ids):
            raise ValueError("Each A/B/C/D label must be exactly one token")
        self.label_ids = [ids[0] for ids in self.label_ids]
        if len(set(self.label_ids)) != len(self.labels):
            raise ValueError("A/B/C/D labels must use distinct token IDs")
        mx.eval(self.model.parameters())
        mx.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.vision_calls = 0
        self.measure_calls = 0

    @classmethod
    def load(cls, model_path: str | Path) -> MinistralLatencyBackend:
        return cls(model_path)

    def _materialize(self, value):
        """Finish all array work in a processor result before timing inference."""
        arrays = []

        def collect(item):
            if isinstance(item, self.mx.array):
                arrays.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    collect(child)

        collect(value)
        if arrays:
            self.mx.eval(*arrays)
        self.mx.synchronize()

    @staticmethod
    def _shapes(value):
        if hasattr(value, "shape"):
            return list(value.shape)
        if isinstance(value, (tuple, list)):
            return [MinistralLatencyBackend._shapes(item) for item in value]
        return None

    def measure(self, prompt: str, images=None) -> dict:
        """Return JSON-compatible timing and token counts; no accuracy metric.

        ``images`` contains in-memory PIL images. Exactly one user prompt is
        evaluated (batch size one). The prompt should ask for A/B/C/D. All
        requests use fresh KV caches and freshly calculated visual features.
        """
        from PIL import Image

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty text prompt is required")
        images = [] if images is None else list(images)
        if len(images) > 4 or any(not isinstance(im, Image.Image) for im in images):
            raise ValueError("Provide at most four in-memory PIL images")
        # The benchmark is serial; lock acquisition is not serving queue latency.
        with self._lock:
            return self._measure(prompt, images)

    def _measure(self, prompt, images):
        mx = self.mx
        mx.synchronize()
        started = time.perf_counter()
        formatted = self._apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=len(images),
            add_generation_prompt=True,
        )
        prefix_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        for label, token_id in zip(self.labels, self.label_ids):
            if self.tokenizer.encode(
                formatted + label, add_special_tokens=False
            ) != prefix_ids + [token_id]:
                raise ValueError(
                    f"Label {label!r} is not one token at the assistant boundary"
                )
        inputs = self._prepare_inputs(
            self.processor,
            images=images or None,
            prompts=formatted,
            image_token_index=self.image_token_id,
        )
        self._materialize(inputs)
        input_ids = inputs["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("This latency backend requires batch size one")
        input_tokens = int(input_ids.shape[-1])
        if input_tokens > 8192:
            raise ValueError("The local latency probe limits inputs to 8192 tokens")
        image_tokens = int(mx.sum(input_ids == self.image_token_id).item())
        pixels_present = inputs.get("pixel_values") is not None
        if images and (not pixels_present or image_tokens == 0):
            raise RuntimeError("The processor dropped the image pixels or tokens")
        if not images and (pixels_present or image_tokens):
            raise RuntimeError("Text-only input unexpectedly contains image data")
        mx.synchronize()
        prepared_at = time.perf_counter()

        # With pixels and no cached_image_features this runs the actual vision
        # tower, patch merger, projector, and insertion into text embeddings.
        features = self.model.get_input_embeddings(**inputs)
        mx.eval(features.inputs_embeds)
        mx.synchronize()
        embedded_at = time.perf_counter()

        # The installed decoder returns final normalized hidden states. Bypass
        # its all-position vocab projection and project the last position only.
        # Each request starts with an empty KV cache; no prefix is reused.
        lm = self.language_model
        cache = lm.make_cache()
        hidden = lm.model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=features.inputs_embeds,
        )[:, -1:, :]
        logits = (
            lm.model.embed_tokens.as_linear(hidden)
            if lm.config.tie_word_embeddings
            else lm.lm_head(hidden)
        )[0, 0, :].astype(mx.float32)
        selected = logits[mx.array(self.label_ids)]
        logprobs = selected - mx.logsumexp(logits)
        probabilities = mx.softmax(selected)
        mx.eval(logprobs, probabilities)
        mx.synchronize()
        logprobs_list = logprobs.tolist()
        probabilities_list = probabilities.tolist()
        if not all(math.isfinite(value) for value in logprobs_list):
            raise RuntimeError("The model produced nonfinite option probabilities")
        finished_at = time.perf_counter()
        self.measure_calls += 1
        self.vision_calls += int(bool(images))
        embedding_ms = (embedded_at - prepared_at) * 1000
        total_ms = (finished_at - started) * 1000
        return {
            "preprocessing_ms": (prepared_at - started) * 1000,
            "embedding_ms": embedding_ms,
            "vision_embedding_ms": embedding_ms if images else 0.0,
            "text_embedding_ms": 0.0 if images else embedding_ms,
            "lm_prefill_ms": (finished_at - embedded_at) * 1000,
            "language_prefill_ms": (finished_at - embedded_at) * 1000,
            "inference_ms": (finished_at - prepared_at) * 1000,
            "total_ms": total_ms,
            "end_to_end_ms": total_ms,
            "input_tokens": input_tokens,
            "image_tokens": image_tokens,
            "non_image_tokens": input_tokens - image_tokens,
            "images": len(images),
            "image_dimensions": [list(image.size) for image in images],
            "pixel_values_shape": self._shapes(inputs.get("pixel_values")),
            "vision_forward_executed": bool(images),
            "generated_tokens": 0,
            "output_tokens": 0,
            "last_position_projection_only": True,
            "prefix_cache": False,
            "vision_feature_cache": False,
            "batch_size": 1,
            "label_token_ids": self.label_ids,
            "label_probability_mass": math.fsum(
                math.exp(value) for value in logprobs_list
            ),
            "option_probabilities": dict(zip(self.labels, probabilities_list)),
        }
