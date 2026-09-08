"""Unit tests for DeepSeek-V4 MHC kernel prewarming."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4MHCPrewarm(CustomTestCase):
    def test_skips_prewarmer_without_sglang_parallel_groups(self):
        model = SimpleNamespace(_mhc_prewarmed_at_load=False)

        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            patch.object(deepseek_v4, "_is_npu", False),
            patch.object(
                deepseek_v4, "model_parallel_is_initialized", return_value=False
            ),
        ):
            deepseek_v4.DeepseekV4ForCausalLM._prewarm_mhc_kernels(model)

        self.assertTrue(model._mhc_prewarmed_at_load)


if __name__ == "__main__":
    unittest.main()
