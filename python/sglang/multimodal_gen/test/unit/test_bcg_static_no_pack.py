"""The static no-pack varlen route: fixed shapes, pad clamped by cu VALUES."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention import layer as layer_module
from sglang.multimodal_gen.runtime.layers.attention.layer import (
    DynamicVarlenMaskMeta,
    USPAttention,
)
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


class TestBCGCaptureRouting(unittest.TestCase):
    """Which (backend, metadata) pairs the BCG wrapper holds inside the graph.

    A pair routed the wrong way either bakes a content-dependent shape into
    the graph or sends a capturable path back to per-layer eager breaks.
    """

    def _route(self, backend, *, meta, opted_in=True, sp_size=1):
        seen = {}

        def capture_path(self, **kwargs):
            seen.update(where="capture", meta=kwargs.get("attn_mask_meta"))

        def break_path(self, **kwargs):
            seen.update(where="break", meta=kwargs.get("attn_mask_meta"))

        with patch(f"{_LAYER}.eager_on_graph", return_value=break_path):
            forward = layer_module._make_breakable_attention_forward(capture_path)
        module = SimpleNamespace(backend=backend, capture_dense_attention=opted_in)
        with (
            patch(f"{_LAYER}.is_in_breakable_cuda_graph", return_value=True),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=sp_size),
        ):
            forward(module, attn_mask_meta=meta)
        return seen

    def test_sdpa_drops_dynamic_meta_and_captures(self):
        seen = self._route(
            AttentionBackendEnum.TORCH_SDPA, meta=DynamicVarlenMaskMeta()
        )
        self.assertEqual(seen["where"], "capture")
        self.assertIsNone(seen["meta"])

    def test_fa_captures_the_static_route(self):
        cu = torch.tensor([0, 4], dtype=torch.int32)
        seen = self._route(AttentionBackendEnum.FA, meta={"static_no_pack_cu": cu})
        self.assertEqual(seen["where"], "capture")

    def test_fa_keeps_the_break_and_the_meta_on_dynamic_varlen(self):
        meta = DynamicVarlenMaskMeta()
        seen = self._route(AttentionBackendEnum.FA, meta=meta)
        self.assertEqual(seen["where"], "break")
        self.assertIs(seen["meta"], meta)

    def test_sequence_parallel_keeps_the_break(self):
        seen = self._route(AttentionBackendEnum.TORCH_SDPA, meta=None, sp_size=2)
        self.assertEqual(seen["where"], "break")

    def test_module_without_opt_in_keeps_the_break(self):
        seen = self._route(
            AttentionBackendEnum.TORCH_SDPA, meta=None, opted_in=False
        )
        self.assertEqual(seen["where"], "break")


if __name__ == "__main__":
    unittest.main()
