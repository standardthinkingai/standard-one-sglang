"""A feature tensor sent many times shares one shared-memory segment (Linux)."""

import ast
import gc
import mmap
import os
import pickle
import sys
import threading
import uuid
import weakref
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not os.path.isdir("/dev/shm"), reason="POSIX shm"
)


SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/managers/mm_utils.py"
CHOSEN = {
    "ShmPointerMMData",
    "_POSIX_SHM_DIR",
    "_reused_shm_features",
    "_reused_shm_features_lock",
    "mark_shm_feature_reused",
    "_release_reused_shm_feature",
    "_link_reused_shm_feature",
    "_wrap_shm_or_inline",
}


def _name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


@pytest.fixture
def mm_utils():
    # Importing mm_utils initializes the GPU runtime; compile only the transport code.
    body = [n for n in ast.parse(SOURCE.read_text()).body if _name(n) in CHOSEN]
    assert {_name(n) for n in body} == CHOSEN
    namespace = {
        "torch": torch, "os": os, "sys": sys, "mmap": mmap, "threading": threading,
        "weakref": weakref, "shared_memory": shared_memory, "Dict": Dict, "Optional": Optional,
        "make_shm_name": lambda kind: f"sgl_shm_{kind}_{os.getpid()}_{uuid.uuid4().hex[:8]}",
        "print_warning_once": lambda message: None,
        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return SimpleNamespace(**namespace)


def _path(name):
    return os.path.join("/dev/shm", name)


def _receive(pointer):
    # The scheduler unpickles (getstate/setstate) and materializes, unlinking the
    # name it received. The class is compiled outside a module, so pickle cannot find it.
    received = type(pointer).__new__(type(pointer))
    received.__setstate__(pickle.loads(pickle.dumps(pointer.__getstate__())))
    return received.materialize()


def test_marked_tensor_is_written_once_and_linked_per_send(mm_utils):
    tensor = torch.arange(4096, dtype=torch.float32).reshape(1, 4, 32, 32)
    mm_utils.mark_shm_feature_reused(tensor)
    first = mm_utils._wrap_shm_or_inline(tensor, precomputed_hash=7)
    second = mm_utils._wrap_shm_or_inline(tensor, precomputed_hash=7)
    segment = mm_utils._reused_shm_features[id(tensor)]
    names = {first.shm_name, second.shm_name, segment.shm_name}
    assert len(names) == 3
    assert os.stat(_path(first.shm_name)).st_ino == os.stat(_path(segment.shm_name)).st_ino
    assert first.precomputed_hash == second.precomputed_hash == 7

    for pointer in (first, second):
        received = _receive(pointer)
        assert torch.equal(received, tensor)
        assert not os.path.exists(_path(pointer.shm_name))
        received.add_(1)  # private copy-on-write: the shared segment is untouched
    assert torch.equal(_receive(mm_utils._wrap_shm_or_inline(tensor)), tensor)

    master = _path(segment.shm_name)
    assert os.path.exists(master)
    del tensor, first, second
    gc.collect()
    assert not os.path.exists(master)


def test_unmarked_tensor_gets_its_own_segment_per_send(mm_utils):
    tensor = torch.ones(1, 3, 8, 8)
    pointers = [mm_utils._wrap_shm_or_inline(tensor) for _ in range(2)]
    inodes = {os.stat(_path(p.shm_name)).st_ino for p in pointers}
    assert len(inodes) == 2
    assert id(tensor) not in mm_utils._reused_shm_features
    for pointer in pointers:
        assert torch.equal(_receive(pointer), tensor)


def test_discarded_link_leaves_other_sends_intact(mm_utils):
    tensor = torch.full((1, 3, 8, 8), 2.0)
    mm_utils.mark_shm_feature_reused(tensor)
    aborted = mm_utils._wrap_shm_or_inline(tensor)
    kept = mm_utils._wrap_shm_or_inline(tensor)
    aborted.close_and_unlink()
    assert torch.equal(_receive(kept), tensor)
