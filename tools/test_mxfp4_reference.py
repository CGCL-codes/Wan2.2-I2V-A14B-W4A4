"""Minimal self-test for the MXFP4 PyTorch reference path."""

from __future__ import annotations
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.nn as nn

from quant.mxfp4 import (
    dequantize_linear_weight_groupwise,
    fp4_e2m1_decode,
    pack_u4,
    quantize_linear_weight_groupwise,
    quantize_linear_weight_groupwise_gptq,
    unpack_u4,
)
from quant.quant_linear import LowRankWeightBranch, QuantLinear, QuantLinearWithBranch


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 0.0, rtol: float = 0.0) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"Tensor mismatch:\nactual={actual}\nexpected={expected}")


def test_fp4_decode_table() -> None:
    codes = torch.arange(16, dtype=torch.uint8)
    actual = fp4_e2m1_decode(codes, out_dtype=torch.float32)
    expected = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    _assert_close(actual, expected)


def test_pack_unpack_roundtrip() -> None:
    codes = torch.tensor([0, 1, 2, 3, 4, 5, 14, 15, 7], dtype=torch.uint8)
    packed = pack_u4(codes)
    restored = unpack_u4(packed, numel=codes.numel())
    if packed.dtype != torch.uint8:
        raise AssertionError(f"Expected packed dtype uint8, got {packed.dtype}")
    if restored.dtype != torch.uint8:
        raise AssertionError(f"Expected restored dtype uint8, got {restored.dtype}")
    if not torch.equal(restored, codes):
        raise AssertionError(f"Roundtrip mismatch:\nactual={restored}\nexpected={codes}")


def test_quant_linear_mxfp4_forward() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(96, 40, bias=True, dtype=torch.float32)
    qlinear = QuantLinear.from_linear(
        linear,
        weight_group_size=32,
        act_group_size=32,
        quantize_input=False,
        skip_input_quant=True,
        compute_dtype="bfloat16",
        scheme="mxfp4",
    )

    x = torch.randn(3, 5, 96, dtype=torch.float32)
    y_ref = linear(x)
    y_q = qlinear(x)

    if y_q.shape != y_ref.shape:
        raise AssertionError(f"Output shape mismatch: got {tuple(y_q.shape)} vs expected {tuple(y_ref.shape)}")
    if qlinear.qweight_packed.dtype != torch.uint8:
        raise AssertionError(f"qweight_packed dtype mismatch: {qlinear.qweight_packed.dtype}")
    if qlinear.w_scales.dtype != torch.uint8:
        raise AssertionError(f"w_scales dtype mismatch: {qlinear.w_scales.dtype}")
    if int(qlinear.w_group_size.item()) != 32:
        raise AssertionError(f"w_group_size mismatch: {int(qlinear.w_group_size.item())}")

    w_dq = qlinear._dequant_weight(device=x.device)
    if not torch.isfinite(w_dq).all():
        raise AssertionError("Dequantized weight contains NaN or Inf")


def test_gptq_mxfp4_packing_compatibility() -> None:
    torch.manual_seed(1)
    w = torch.randn(17, 65, dtype=torch.float32) * 0.2
    h = torch.eye(w.shape[1], dtype=torch.float32)
    q = quantize_linear_weight_groupwise_gptq(w, hessian=h, group_size=32)
    w_dq = dequantize_linear_weight_groupwise(
        q["qweight_packed"],
        q["scales"],
        in_features=w.shape[1],
        group_size=32,
        out_dtype=torch.float32,
        scheme="mxfp4",
    )
    if w_dq.shape != w.shape:
        raise AssertionError(f"GPTQ dequant shape mismatch: {tuple(w_dq.shape)} vs {tuple(w.shape)}")
    if not torch.isfinite(w_dq).all():
        raise AssertionError("GPTQ dequantized weight contains NaN or Inf")
    if q["scales"].dtype != torch.uint8:
        raise AssertionError(f"GPTQ scales dtype mismatch: {q['scales'].dtype}")


def test_gptq_mxfp4_not_obviously_worse_than_rtn() -> None:
    torch.manual_seed(2)
    w = torch.randn(24, 96, dtype=torch.float32) * 0.15
    x = torch.randn(384, 96, dtype=torch.float32)
    h = x.T @ x / float(x.shape[0])
    q_rtn = quantize_linear_weight_groupwise(w, group_size=32, scheme="mxfp4")
    q_gptq = quantize_linear_weight_groupwise_gptq(w, hessian=h, group_size=32, block_size=64)
    w_rtn = dequantize_linear_weight_groupwise(
        q_rtn["qweight_packed"], q_rtn["scales"], w.shape[1], 32, out_dtype=torch.float32, scheme="mxfp4"
    )
    w_gptq = dequantize_linear_weight_groupwise(
        q_gptq["qweight_packed"], q_gptq["scales"], w.shape[1], 32, out_dtype=torch.float32, scheme="mxfp4"
    )
    err_rtn = torch.linalg.norm((w - w_rtn) @ x.T)
    err_gptq = torch.linalg.norm((w - w_gptq) @ x.T)
    if err_gptq > err_rtn * 1.25:
        raise AssertionError(f"GPTQ output error is unexpectedly worse: gptq={err_gptq} rtn={err_rtn}")


def test_svdquant_residual_gptq_shape_and_range() -> None:
    torch.manual_seed(3)
    w_ref = torch.randn(32, 96, dtype=torch.float32) * 0.1
    x = torch.randn(256, 96, dtype=torch.float32)
    h = x.T @ x / float(x.shape[0])
    branch = LowRankWeightBranch.from_weight(w_ref, rank=4, compute_dtype=torch.bfloat16)
    w_lowrank = branch.effective_weight()
    w_main = w_ref - w_lowrank
    q = quantize_linear_weight_groupwise_gptq(w_main, hessian=h, group_size=32)
    w_main_dq = dequantize_linear_weight_groupwise(
        q["qweight_packed"], q["scales"], w_ref.shape[1], 32, out_dtype=torch.float32, scheme="mxfp4"
    )
    w_total = w_lowrank + w_main_dq
    if w_total.shape != w_ref.shape:
        raise AssertionError(f"SVDQuant total shape mismatch: {tuple(w_total.shape)}")
    if not torch.isfinite(w_total).all():
        raise AssertionError("SVDQuant GPTQ reconstructed weight contains NaN or Inf")


def test_quantlinear_fallback_and_rank_paths() -> None:
    torch.manual_seed(4)
    linear = nn.Linear(64, 16, bias=False, dtype=torch.float32)
    q_rtn = QuantLinearWithBranch.from_linear_svdquant(
        linear,
        rank=0,
        weight_group_size=32,
        act_group_size=32,
        compute_dtype="bfloat16",
    )
    if q_rtn.quant_linear.quant_method != "rtn_mxfp4":
        raise AssertionError(f"Expected RTN fallback, got {q_rtn.quant_linear.quant_method}")

    h = torch.eye(64, dtype=torch.float32)
    q_gptq = QuantLinearWithBranch.from_linear_svdquant(
        linear,
        rank=4,
        weight_group_size=32,
        act_group_size=32,
        compute_dtype="bfloat16",
        gptq_hessian=h,
    )
    if q_gptq.quant_linear.quant_method != "gptq_mxfp4":
        raise AssertionError(f"Expected GPTQ method, got {q_gptq.quant_linear.quant_method}")
    if q_gptq.branch is None:
        raise AssertionError("Expected low-rank branch for rank > 0")


def main() -> None:
    test_fp4_decode_table()
    test_pack_unpack_roundtrip()
    test_quant_linear_mxfp4_forward()
    test_gptq_mxfp4_packing_compatibility()
    test_gptq_mxfp4_not_obviously_worse_than_rtn()
    test_svdquant_residual_gptq_shape_and_range()
    test_quantlinear_fallback_and_rank_paths()
    print("MXFP4 reference self-test passed.")


if __name__ == "__main__":
    main()
