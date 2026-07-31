"""Unit tests for tensor-backed sampling-mask packing."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def test_sampling_mask_is_packed_as_device_tensors():
    probs = torch.tensor(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    sampling_info = SimpleNamespace(
        sampling_mask_max_top_k=3,
        top_ks=torch.tensor([3, 2], dtype=torch.int32),
        top_ps=torch.tensor([0.6, 1.0], dtype=torch.float32),
        min_ps=torch.zeros(2, dtype=torch.float32),
        need_min_p_sampling=False,
    )
    sampled_tokens = torch.tensor([1, 0], dtype=torch.int64)
    output = LogitsProcessorOutput(next_token_logits=None)

    Sampler._attach_sampling_mask_to_output(
        None, output, sampling_info, sampled_tokens, probs
    )

    assert isinstance(output.next_token_sampling_mask_idx, torch.Tensor)
    assert isinstance(output.next_token_sampling_mask_len, torch.Tensor)
    assert isinstance(output.next_token_sampling_logprobs, torch.Tensor)
    assert output.next_token_sampling_mask_len.tolist() == [2, 3]
    assert output.next_token_sampling_mask_idx[0, :2].tolist() == [0, 1]
    assert output.next_token_sampling_mask_idx[1, :3].tolist() == [3, 2, 0]
    assert output.next_token_sampling_logprobs.tolist() == pytest.approx(
        [
            torch.log(torch.tensor(0.3 / 0.7)).item(),
            torch.log(torch.tensor(0.1 / 0.8)).item(),
        ]
    )
