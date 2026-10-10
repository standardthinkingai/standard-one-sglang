"""Exercise real model constructors without loading CUDA or the SGLang package.

The construction methods are compiled from their source files into a small
dependency harness. This catches a repeated ``make_layers`` call, which would
consume OffloaderV1's byte budget on decoder layers that are then discarded.
"""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


SG_SRT = Path(__file__).resolve().parents[3] / "python" / "sglang" / "srt"


def _constructor_class(path: Path, class_name: str) -> ast.ClassDef:
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    constructor = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    isolated = copy.deepcopy(cls)
    isolated.body = [copy.deepcopy(constructor)]
    return isolated


class TestMinistral3DecoderConstruction(unittest.TestCase):
    def _harness(self, *, pp_rank: int, pp_size: int, num_layers: int):
        constructed = []
        make_layers_calls = []
        offloaded = []

        class Module:
            def __init__(self, *args, **kwargs):
                pass

        class LlamaDecoderLayer(Module):
            def __init__(
                self, *, config, quant_config, layer_id, start_layer, prefix
            ):
                constructed.append((type(self).__name__, layer_id, start_layer, prefix))

        class Ministral3DecoderLayer(LlamaDecoderLayer):
            pass

        def get_pp_indices(total, rank, size):
            return total * rank // size, total * (rank + 1) // size

        def make_layers(total, layer_fn, *, pp_rank, pp_size, prefix):
            make_layers_calls.append((total, pp_rank, pp_size, prefix))
            start, end = get_pp_indices(total, pp_rank, pp_size)
            modules = []
            for index in range(start, end):
                module = layer_fn(index, f"{prefix}.{index}")
                modules.append(module)
                offloaded.append(module)
            return modules, start, end

        pp_group = SimpleNamespace(
            rank_in_group=pp_rank,
            world_size=pp_size,
            is_first_rank=pp_rank == 0,
            is_last_rank=pp_rank == pp_size - 1,
        )
        namespace = {
            "nn": SimpleNamespace(Module=Module),
            "LlamaDecoderLayer": LlamaDecoderLayer,
            "Ministral3DecoderLayer": Ministral3DecoderLayer,
            "get_parallel": lambda: SimpleNamespace(pp_group=pp_group),
            "get_pp_indices": get_pp_indices,
            "make_layers": make_layers,
            "VocabParallelEmbedding": Module,
            "PPMissingLayer": Module,
            "RMSNorm": Module,
            "add_prefix": lambda name, prefix: f"{prefix}.{name}" if prefix else name,
        }
        classes = [
            _constructor_class(SG_SRT / "models" / "llama.py", "LlamaModel"),
            _constructor_class(
                SG_SRT / "models" / "ministral3.py", "Ministral3Model"
            ),
        ]
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *classes,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(tree), "model_constructor_harness", "exec"),
            namespace,
        )
        config = SimpleNamespace(
            pad_token_id=0,
            vocab_size=128,
            hidden_size=16,
            num_hidden_layers=num_layers,
            rms_norm_eps=1e-5,
        )
        return namespace, config, constructed, make_layers_calls, offloaded

    def test_ministral_layers_are_built_and_offloaded_once(self):
        for pp_rank, pp_size, num_layers in ((0, 1, 6), (1, 2, 6)):
            with self.subTest(pp_rank=pp_rank, pp_size=pp_size):
                namespace, config, constructed, calls, offloaded = self._harness(
                    pp_rank=pp_rank, pp_size=pp_size, num_layers=num_layers
                )
                model = namespace["Ministral3Model"](
                    config, prefix="language_model.model"
                )
                start = num_layers * pp_rank // pp_size
                end = num_layers * (pp_rank + 1) // pp_size
                self.assertEqual(
                    calls, [(num_layers, pp_rank, pp_size, "model.layers")]
                )
                self.assertEqual((model.start_layer, model.end_layer), (start, end))
                self.assertEqual(len(constructed), end - start)
                self.assertEqual(len(offloaded), end - start)
                self.assertTrue(all(layer in model.layers for layer in offloaded))
                self.assertEqual(
                    constructed,
                    [
                        (
                            "Ministral3DecoderLayer",
                            index,
                            start,
                            f"model.layers.{index}",
                        )
                        for index in range(start, end)
                    ],
                )

    def test_default_llama_layers_still_use_llama_decoder(self):
        namespace, config, constructed, calls, offloaded = self._harness(
            pp_rank=0, pp_size=1, num_layers=4
        )
        model = namespace["LlamaModel"](config)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(offloaded), 4)
        self.assertEqual(len(model.layers), 4)
        self.assertTrue(all(kind == "LlamaDecoderLayer" for kind, *_ in constructed))


if __name__ == "__main__":
    unittest.main()
