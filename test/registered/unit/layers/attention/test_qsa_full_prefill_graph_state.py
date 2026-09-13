import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestQSAFullPrefillGraphState(unittest.TestCase):
    def test_uses_phase_private_metadata_and_survives_decode_initialization(self):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.device = torch.device("cpu")
        backend.max_context_len = 4096
        backend.compress_ratio = 4
        backend.token_to_kv_pool = SimpleNamespace(qsa_compressed_page_size=64)

        backend.init_full_prefill_cuda_graph_state(
            max_bs=8, max_num_tokens=16_384
        )
        self.assertEqual(backend._full_prefill_cuda_graph_metadata, {})

        backend.init_cuda_graph_state(max_bs=2, max_num_tokens=2)

        self.assertEqual(backend._graph_seq_lens.shape, (2,))
        self.assertEqual(backend._full_prefill_cuda_graph_metadata, {})


if __name__ == "__main__":
    unittest.main()
