import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _load_base_module(monkeypatch):
    """Load the focused module without importing the full SGLang runtime."""
    memory_pool = ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.ReqToTokenPool = object
    forward_batch_info = ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_batch_info.ForwardBatch = object
    monkeypatch.setitem(sys.modules, memory_pool.__name__, memory_pool)
    monkeypatch.setitem(sys.modules, forward_batch_info.__name__, forward_batch_info)

    path = Path(__file__).parents[3] / "python/sglang/srt/state_capturer/base.py"
    spec = importlib.util.spec_from_file_location(
        "state_capturer_base_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_overlap_output_snapshots_shared_topk_buffer(monkeypatch):
    BaseTopkCapturer = _load_base_module(monkeypatch).BaseTopkCapturer
    capturer = BaseTopkCapturer.__new__(BaseTopkCapturer)
    capturer.topk_size = 2
    capturer.host_cache = object()
    capturer.device_cache = SimpleNamespace(
        buffer=torch.arange(18, dtype=torch.int32).reshape(3, 3, 2)
    )
    forward_batch = SimpleNamespace(out_cache_loc=torch.tensor([4, 5]))

    output = capturer.on_forward_end(
        forward_batch=forward_batch,
        can_run_graph=False,
        cuda_graph_batch=None,
        no_copy_to_cpu=True,
    )
    expected = output.topk.clone()

    capturer.device_cache.buffer.fill_(-1)

    torch.testing.assert_close(output.topk, expected, rtol=0, atol=0)
