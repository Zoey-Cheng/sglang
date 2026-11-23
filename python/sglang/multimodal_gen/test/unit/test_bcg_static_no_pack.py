"""The static no-pack varlen route: fixed shapes, pad clamped by cu VALUES."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


def _attn(backend: AttentionBackendEnum) -> USPAttention:
    obj = USPAttention.__new__(USPAttention)
    obj.skip_sequence_parallel = False
    obj.softmax_scale = 0.5
    obj.backend = backend
    obj.sp_attention_mode = "ulysses"
    obj.allow_cudnn_sdp = False
    obj.causal = False
    return obj


def _single_rank_patches():
    return (
        patch(
            f"{_LAYER}.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=1),
    )


class TestStaticNoPackVarlen(unittest.TestCase):
    def test_fa_gets_full_buffers_and_cu_values(self):
        obj = _attn(AttentionBackendEnum.FA)
        seq, valid, heads, dim = 8, 5, 2, 4
        q = torch.randn(1, seq, heads, dim)
        cu = torch.tensor([0, valid], dtype=torch.int32)
        seen = {}

        def fake_varlen(**kw):
            seen.update(kw)
            return torch.zeros(seq, heads, dim)

        ctx, sp = _single_rank_patches()
        with ctx, sp, patch(
            f"{_LAYER}.flash_attn_varlen_func", side_effect=fake_varlen
        ):
            out = obj.forward(q, q, q, attn_mask_meta={"static_no_pack_cu": cu})

        self.assertEqual(out.shape, (1, seq, heads, dim))
        self.assertEqual(seen["q"].shape, (seq, heads, dim))
        self.assertIs(seen["cu_seqlens_q"], cu)
        self.assertEqual(seen["max_seqlen_q"], seq)

    def test_non_fa_backend_matches_truncated_sdpa_on_valid_rows(self):
        obj = _attn(AttentionBackendEnum.TORCH_SDPA)
        seq, valid, heads, dim = 8, 5, 2, 4
        torch.manual_seed(0)
        q = torch.randn(1, seq, heads, dim)
        k = torch.randn(1, seq, heads, dim)
        v = torch.randn(1, seq, heads, dim)
        cu = torch.tensor([0, valid], dtype=torch.int32)

        ctx, sp = _single_rank_patches()
        with ctx, sp:
            out = obj.forward(q, k, v, attn_mask_meta={"static_no_pack_cu": cu})

        ref = torch.nn.functional.scaled_dot_product_attention(
            q[:, :valid].transpose(1, 2),
            k[:, :valid].transpose(1, 2),
            v[:, :valid].transpose(1, 2),
            scale=obj.softmax_scale,
        ).transpose(1, 2)
        self.assertEqual(out.shape, (1, seq, heads, dim))
        torch.testing.assert_close(out[:, :valid], ref, rtol=1e-5, atol=1e-5)

    def test_mask_alongside_static_meta_raises(self):
        obj = _attn(AttentionBackendEnum.FA)
        q = torch.randn(1, 4, 2, 4)
        ctx, sp = _single_rank_patches()
        with ctx, sp, self.assertRaises(ValueError):
            obj.forward(
                q,
                q,
                q,
                attn_mask=torch.ones(1, 4, dtype=torch.bool),
                attn_mask_meta={
                    "static_no_pack_cu": torch.tensor([0, 4], dtype=torch.int32)
                },
            )


if __name__ == "__main__":
    unittest.main()
