import unittest

from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMambaSharedReadFence(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(MambaAttnBackendBase)

    def test_decode_reads_end_after_graph_replay(self):
        self.assertIs(
            self.backend.shared_read_ends(ForwardMode.DECODE),
            SharedReadEnds.POST_REPLAY,
        )

    def test_other_modes_keep_default_boundaries(self):
        self.assertIs(
            self.backend.shared_read_ends(ForwardMode.TARGET_VERIFY),
            SharedReadEnds.IN_REPLAY,
        )
        self.assertIs(
            self.backend.shared_read_ends(ForwardMode.EXTEND),
            SharedReadEnds.UNKNOWN,
        )


if __name__ == "__main__":
    unittest.main()
