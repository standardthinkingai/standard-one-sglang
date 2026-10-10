import copy
import logging
import math
from typing import Any, List, Optional, Union

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.pixtral import (
    PixtralForConditionalGeneration,
    PixtralVisionModel,
)
from sglang.srt.multimodal.decision_images import rebuild_decision_image_ids
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from transformers import PreTrainedTokenizerBase
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)

logger = logging.getLogger(__name__)


class PixtralProcessor(BaseMultimodalProcessor):
    models = [PixtralVisionModel, PixtralForConditionalGeneration]
    gpu_image_decode = False  # Pixtral processes loaded image as PIL image explicitly

    PAD_TOKEN = "<pad>"
    DEFAULT_IMAGE_TOKEN = "[IMG]"
    # Retained processed images per tokenizer process in decision mode; a
    # 1540 px image is about 28 MiB of float32 pixels.
    DECISION_IMAGE_CACHE_MB = 1024

    @property
    def auto_mm_preprocess_cache_size_mb(self) -> int:
        # A decision request sends one evaluation per question and option
        # rotation, all with the same images.
        if getattr(self.server_args, "decision_readout_path", None):
            return self.DECISION_IMAGE_CACHE_MB
        return 0

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self._decision_readout_enabled = bool(
            getattr(server_args, "decision_readout_path", None)
        )
        if self._decision_readout_enabled and not self._shares_decision_images():
            logger.warning(
                "Decision images are preprocessed once per evaluation; set "
                "--mm-preprocess-cache-size-mb above 0 with CPU feature transport "
                "to preprocess them once per request."
            )
        self.IM_TOKEN_ID = getattr(
            hf_config, "image_token_index", PixtralVisionModel.DEFAULT_IMAGE_TOKEN_ID
        )

        self.vision_config = hf_config.vision_config
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size

        # spatial_merge_size may live on vision_config (Mistral native) or
        # on the top-level config (HF native Mistral3Config).
        self._spatial_merge_size = getattr(
            self.vision_config,
            "spatial_merge_size",
            getattr(hf_config, "spatial_merge_size", 1),
        )
        self._effective_patch_size = self.patch_size * self._spatial_merge_size

        self._processor.patch_size = self.patch_size
        if self._spatial_merge_size > 1:
            self._processor.spatial_merge_size = self._spatial_merge_size

        tokenizer = (
            _processor
            if isinstance(_processor, PreTrainedTokenizerBase)
            else _processor.tokenizer
        )
        self.image_token = getattr(_processor, "image_token", self.DEFAULT_IMAGE_TOKEN)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)
        tokenizer.add_special_tokens(
            {
                "pad_token": getattr(hf_config, "pad_token", self.PAD_TOKEN),
            }
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        decision_ids = None
        if self._decision_readout_enabled and image_data:
            # Native Mistral control tokens do not roundtrip through decode +
            # encode. Process images with a marker-only prompt, then splice the
            # actual patch rows into the untouched native chat before offsets
            # and transport buffers are built.
            if (
                not isinstance(input_text, list)
                or not input_text
                or any(type(token) is not int or token < 0 for token in input_text)
                or input_text.count(self.IM_TOKEN_ID) != len(image_data)
            ):
                raise ValueError("decision images require one native placeholder per image")
            decision_ids = list(input_text)
            if self._shares_decision_images():
                return await self._process_shared_decision_images(
                    image_data, decision_ids
                )
            input_text = self.image_token * len(image_data)
        mm_data = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=True,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            mm_data,
            self.mm_tokens,
            **({"decision_input_ids": decision_ids} if decision_ids is not None else {}),
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.IM_TOKEN_ID,
        )

    def _shares_decision_images(self) -> bool:
        # CUDA IPC leases are single-use, so device-resident features are
        # processed per evaluation.
        return (
            self.mm_preprocess_cache.enabled and not self.keep_mm_features_on_device
        )

    async def _process_shared_decision_images(self, image_data, decision_ids):
        """Process a request's images once and splice them into each evaluation.

        Evaluations of one decision request carry identical images and differ
        only in their native text IDs. Concurrent evaluations join one
        computation; later ones read the bounded LRU entry.
        """
        images = tuple(image_data)
        lookup = await self.mm_preprocess_cache.get_or_compute(
            ("decision-images", images),
            lambda: self._decision_image_bundle(images),
            size_bytes=lambda bundle: self._decision_bundle_bytes(images, bundle),
        )
        items, processed_ids = lookup.value
        break_token = getattr(self._processor, "image_break_token", "[IMG_BREAK]")
        end_token = getattr(self._processor, "image_end_token", "[IMG_END]")
        rebuilt, offsets = rebuild_decision_image_ids(
            decision_ids,
            processed_ids,
            [item.offsets for item in items],
            image_token_id=self.IM_TOKEN_ID,
            image_break_id=self._tokenizer.convert_tokens_to_ids(break_token),
            image_end_id=self._tokenizer.convert_tokens_to_ids(end_token),
        )
        evaluation_items = []
        for item, item_offsets in zip(items, offsets):
            # Transport rebinds `feature` on this copy; the cached item keeps
            # its tensor for the request's other evaluations.
            evaluation_item = copy.copy(item)
            evaluation_item.offsets = item_offsets
            evaluation_item.model_specific_data = dict(item.model_specific_data)
            evaluation_items.append(evaluation_item)
        return MultimodalProcessorOutput(
            mm_items=evaluation_items,
            input_ids=rebuilt,
            im_token_id=self.IM_TOKEN_ID,
        )

    async def _decision_image_bundle(self, images):
        """Image items with offsets into their expanded patch blocks alone."""
        mm_data = await self.load_mm_data(
            prompt=self.image_token * len(images),
            multimodal_tokens=self.mm_tokens,
            image_data=list(images),
            return_text=True,
        )
        # Splicing into bare placeholders validates the patch blocks before
        # offsets and transport, as for a single evaluation.
        items, processed_ids, _ = await self.process_and_combine_mm_data_async(
            mm_data,
            self.mm_tokens,
            decision_input_ids=[self.IM_TOKEN_ID] * len(images),
        )
        if len(items) != len(images) or not all(
            item.modality == Modality.IMAGE for item in items
        ):
            raise ValueError("decision image placeholders, blocks, and features must align")
        from sglang.srt.managers.mm_utils import mark_shm_feature_reused

        for item in items:
            # Hash once here so the scheduler neither rehashes each evaluation's
            # pixels nor misses its embedding cache between evaluations.
            item.set_pad_value()
            features = item.feature if isinstance(item.feature, list) else [item.feature]
            for feature in features:
                if isinstance(feature, torch.Tensor) and feature.is_cpu:
                    mark_shm_feature_reused(feature)
        return items, processed_ids.tolist()

    @staticmethod
    def _decision_bundle_bytes(images, bundle) -> Optional[int]:
        from sglang.srt.multimodal.cache.preprocess_cache import (
            estimate_cache_size_bytes,
        )

        items, processed_ids = bundle
        features = estimate_cache_size_bytes([item.feature for item in items])
        if features is None:
            return None
        # The key holds the request's data URLs until the entry is evicted.
        return features + sum(len(image) for image in images) + 8 * len(processed_ids)

    def _process_and_collect_mm_items(self, *args, decision_input_ids=None, **kwargs):
        items, input_ids, result = super()._process_and_collect_mm_items(*args, **kwargs)
        if decision_input_ids is None:
            return items, input_ids, result
        if any(item.offsets is not None for item in items):
            raise ValueError("decision image processor must derive offsets from native IDs")
        break_token = getattr(self._processor, "image_break_token", "[IMG_BREAK]")
        end_token = getattr(self._processor, "image_end_token", "[IMG_END]")
        rebuilt, _ = rebuild_decision_image_ids(
            decision_input_ids,
            input_ids.tolist(),
            None,
            image_token_id=self.IM_TOKEN_ID,
            image_break_id=self._tokenizer.convert_tokens_to_ids(break_token),
            image_end_id=self._tokenizer.convert_tokens_to_ids(end_token),
        )
        return (
            items,
            torch.tensor(rebuilt, dtype=input_ids.dtype, device=input_ids.device),
            result,
        )

    def _postprocess_mm_items_before_transport(
        self,
        mm_items: List[MultimodalDataItem],
        *,
        images: Optional[List[Any]],
    ) -> List[MultimodalDataItem]:
        if not images or len(images) <= 1:
            return mm_items

        image_items = [item for item in mm_items if item.modality == Modality.IMAGE]
        if len(image_items) == len(images):
            return mm_items
        if len(image_items) != 1:
            raise ValueError(
                "Pixtral multi-image processing expected one bundled IMAGE item or "
                f"{len(images)} split items, but found {len(image_items)}"
            )

        old_item = image_items[0]
        all_offsets = old_item.offsets
        old_feature = old_item.feature
        old_image_sizes = old_item.model_specific_data.get("image_sizes")
        image_nrows = self._get_image_nrows(images)
        if old_feature is None or len(old_feature) != len(image_nrows):
            raise ValueError(
                "Pixtral multi-image feature count does not match the number of "
                f"images: features={0 if old_feature is None else len(old_feature)}, "
                f"images={len(image_nrows)}"
            )
        if all_offsets is None or sum(image_nrows) != len(all_offsets):
            raise ValueError(
                "Pixtral image patch rows do not match the computed offsets: "
                f"rows={sum(image_nrows)}, "
                f"offsets={0 if all_offsets is None else len(all_offsets)}"
            )

        split_items = [item for item in mm_items if item.modality != Modality.IMAGE]
        offset_idx = 0
        for image_idx, num_rows in enumerate(image_nrows):
            item_offsets = all_offsets[offset_idx : offset_idx + num_rows]
            offset_idx += num_rows
            new_item = copy.copy(old_item)
            new_item.feature = old_feature[image_idx : image_idx + 1]
            new_item.offsets = item_offsets
            new_item.model_specific_data = copy.copy(old_item.model_specific_data)
            if old_image_sizes is not None:
                new_item.model_specific_data["image_sizes"] = old_image_sizes[
                    image_idx : image_idx + 1
                ]
            new_item.hash = None
            new_item.pad_value = None
            split_items.append(new_item)
        if offset_idx != len(all_offsets):
            raise ValueError(
                "Pixtral multi-image split did not consume every offset: "
                f"consumed={offset_idx}, offsets={len(all_offsets)}"
            )
        return split_items

    def decision_image_tokens(self, image: Union[str, bytes]) -> int:
        """Prompt tokens one decision image expands into, read from its header.

        Mirrors this processor's expansion without decoding pixels: the longest
        edge is scaled down to image_size, the image is covered by patches of
        the effective patch size, and each patch row ends in [IMG_BREAK] or
        [IMG_END]. Raises ValueError for anything but a readable image, given
        as a base64 PNG, JPEG or WebP data URL or as raw bytes.
        """
        import base64
        import binascii
        import io
        import re

        from PIL import Image

        if isinstance(image, str):
            match = re.fullmatch(
                r"data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})", image
            )
            if match is None:
                raise ValueError("decision images must be base64 data URLs")
            try:
                image = base64.b64decode(match.group(1), validate=True)
            except binascii.Error as error:
                raise ValueError("decision image is not valid base64") from error
        if not isinstance(image, bytes) or not image:
            raise ValueError("decision images must be nonempty data")
        try:
            with Image.open(io.BytesIO(image)) as opened:
                width, height = opened.size
        except Exception as error:
            raise ValueError("decision image cannot be read") from error
        ratio = max(width / self.image_size, height / self.image_size)
        if ratio > 1:
            width = int(math.floor(width / ratio))
            height = int(math.floor(height / ratio))
        if width < 1 or height < 1:
            raise ValueError("decision image has no pixels")
        rows, columns = _get_pixtral_hf_num_image_tokens(
            (height, width),
            (self._effective_patch_size, self._effective_patch_size),
        )
        return rows * (columns + 1)

    def _get_image_nrows(self, images: List[Any]) -> List[int]:
        image_nrows = []
        for image in images:
            width, height = image.size
            ratio = max(width / self.image_size, height / self.image_size)
            if ratio > 1:
                width = int(math.floor(width / ratio))
                height = int(math.floor(height / ratio))
            num_rows, _ = _get_pixtral_hf_num_image_tokens(
                (height, width),
                (self._effective_patch_size, self._effective_patch_size),
            )
            image_nrows.append(num_rows)
        return image_nrows
