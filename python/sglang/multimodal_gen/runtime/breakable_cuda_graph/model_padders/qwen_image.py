# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Qwen-Image breakable CUDA graph (BCG) prompt padding.

Qwen-Image / Qwen-Image-Edit carry text length on dim 1 of
``encoder_hidden_states`` and a separate ``freqs_cis`` text-rope cache plus
``txt_seq_lens``; they may not pass an explicit prompt mask, so this padder
synthesizes one. Registered with the base denoising stage's padder registry.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph import (
    prompt_padding as bcg_utils,
)


def is_qwen_transformer(current_model: Any, call_kwargs: dict) -> bool:
    return (
        bcg_utils.transformer_class_name_matches(current_model, "qwen")
        and "txt_seq_lens" in call_kwargs
        and "freqs_cis" in call_kwargs
    )


def pad_qwen_prompt_kwargs(
    call_kwargs: dict, current_model: Any, buckets: tuple[int, ...]
) -> dict:
    ehs = call_kwargs.get("encoder_hidden_states")
    ehs_tensor = bcg_utils.first_tensor(ehs)
    if not torch.is_tensor(ehs_tensor) or ehs_tensor.dim() < 2:
        return call_kwargs

    seq = ehs_tensor.shape[1]
    bucket = bcg_utils.select_text_bucket(seq, buckets)
    if bucket is None:
        return call_kwargs

    out = dict(call_kwargs)
    if seq < bucket:
        out["encoder_hidden_states"] = bcg_utils.pad_nested_dim(
            ehs, dim=1, source=seq, target=bucket
        )
        if (
            "encoder_hidden_states_2" in out
            and out["encoder_hidden_states_2"] is not None
        ):
            out["encoder_hidden_states_2"] = bcg_utils.pad_nested_dim(
                out["encoder_hidden_states_2"], dim=1, source=seq, target=bucket
            )

    mask = out.get("encoder_hidden_states_mask")
    if mask is None:
        mask = torch.ones(
            ehs_tensor.shape[:2],
            device=ehs_tensor.device,
            dtype=torch.bool,
        )
    if mask is not None:
        out["encoder_hidden_states_mask"] = bcg_utils.pad_nested_dim(
            mask, dim=1, source=seq, target=bucket
        )

    if "encoder_attention_mask" in out and out["encoder_attention_mask"] is not None:
        out["encoder_attention_mask"] = bcg_utils.pad_nested_dim(
            out["encoder_attention_mask"], dim=1, source=seq, target=bucket
        )

    # the real and the complex RoPE caches both carry the text length on dim 0
    for key in ("freqs_cis", "freqs_complex"):
        freqs = out.get(key)
        if isinstance(freqs, (tuple, list)) and len(freqs) == 2:
            img_cache, txt_cache = freqs
            txt_cache = bcg_utils.pad_nested_dim(
                txt_cache, dim=0, source=seq, target=bucket
            )
            out[key] = type(freqs)((img_cache, txt_cache))

    raw_txt_lens = call_kwargs.get("txt_seq_lens")
    if (
        isinstance(raw_txt_lens, (list, tuple))
        and len(raw_txt_lens) == 1
        and ehs_tensor.shape[0] == 1
        # A caller-supplied mask may carry interior holes the length cannot
        # express; only the mask this padder synthesizes from the length is
        # guaranteed to be a contiguous valid prefix.
        and call_kwargs.get("encoder_hidden_states_mask") is None
        # A caller-prepadded text buffer (seq > valid length) must keep the
        # mask route: the cu value would exclude rows the miss path attends.
        and int(raw_txt_lens[0]) == seq
        and bcg_utils.transformer_class_name_matches(
            current_model, "qwenimagetransformer2d"
        )
    ):
        # Carry the valid text length as a device VALUE: the forward then
        # stays shape-static and free of nonzero(), so the attention itself
        # can be captured (see static_no_pack_cu in USPAttention).
        out["txt_bucket_tail_cu"] = torch.tensor(
            [int(raw_txt_lens[0])], dtype=torch.int32, device=ehs_tensor.device
        )

    # the mask alone marks the valid text: a bucketed length would make the
    # eager forward of a missed graph attend the pad rows
    out["txt_seq_lens"] = None
    return out


bcg_utils.register_prompt_padder(is_qwen_transformer, pad_qwen_prompt_kwargs)
