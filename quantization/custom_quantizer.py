"""
Custom-model CosConv layer quantization (float-in / float-out).

Motivation
----------
The original toolbox quantization pipeline targets the built-in (CosCNN / GroupCosCNN) topology
and performs end-to-end activation quantization. For arbitrary user-defined models (e.g., ResNet),
we instead quantize ONLY the CosConv layers, and keep the surrounding network in float.

Design
------
- Detect modules of type `models.model.CosConvLayer` inside an arbitrary `torch.nn.Module`.
- For each CosConvLayer:
    * Quantize `w` to discrete levels and build cosine LUT (same idea as the original codebase).
    * Quantize `A`.
    * Precompute a dequantized float convolution weight tensor using LUT, so the layer remains:
        float input  -> conv1d(weight_float) -> float output
  This preserves compatibility with residual connections / arbitrary graph structure.

The quantization math is intentionally aligned with `quantization/quantizer.py`:
- w quantization: q in [1, 2*Qmax_w+1]
- LUT stores round(cos(w*x) * Qmax_wx)
- A quantization: round(A * Qmax_A / Amax)

Note
----
This module does NOT require the simple sequential "CosCNN" structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # When imported from the project root
    from models.model import CosConvLayer
except Exception:  # pragma: no cover
    # Fallback for other execution contexts
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.model import CosConvLayer



def _is_cosconv_like(m: nn.Module) -> bool:
    """Heuristic to detect CosConv modules in user-defined models.

    Primary: isinstance(m, CosConvLayer) (toolbox built-in).
    Fallback: user-defined class named 'CosConv' / 'CosConvLayer' with the same essential attributes.
    """
    if isinstance(m, CosConvLayer):
        return True
    cls_name = m.__class__.__name__.lower()
    if cls_name not in {"cosconv", "cosconvlayer"}:
        return False
    required = ["A", "w", "x_buffer", "filter_length", "num_stride", "groups"]
    return all(hasattr(m, k) for k in required)

def getQ(n: int) -> int:
    """Signed symmetric quantization max value: Q = (2^n - 1)//2."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return (2 ** n - 1) // 2


@dataclass
class CosConvQuantConfig:
    """Bit-width configuration for per-layer CosConv quantization."""
    nbit_w: int = 12
    nbit_wx: int = 8
    nbit_A: int = 8
    # For numeric safety
    eps: float = 1e-12


class QuantizedCosConvFloatIO(nn.Module):
    """
    Quantized CosConv layer with float input / float output.

    It precomputes an *approximate* float convolution kernel using a quantized cosine LUT.
    """
    def __init__(
        self,
        *,
        filter_length: int,
        stride: int,
        groups: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.filter_length = int(filter_length)
        self.stride = int(stride)
        self.groups = int(groups)
        if self.groups < 1:
            raise ValueError(f"groups must be >= 1, got {self.groups}")
        self.padding = self.filter_length // 2

        # Store precomputed float weight kernel
        self.register_buffer("weight_qfloat", weight.detach().to(torch.float32))

        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias_qfloat", bias.detach().to(torch.float32))
            self.bias = "bias_qfloat"

        # Optional: store metadata for reproducibility/debug
        self.meta: Dict[str, Any] = meta or {}

    @staticmethod
    def _build_cosine_lut(x: np.ndarray, qcfg: CosConvQuantConfig, scale_w: float) -> Tuple[np.ndarray, float, int]:
        """
        Build cosine LUT aligned with quantizer.py.

        Returns:
            lut_int: int LUT of shape [num_levels, L]
            scale_wx: scale used for LUT values
            qmax_w: qmax_w
        """
        qmax_w = getQ(qcfg.nbit_w)
        qmax_wx = getQ(qcfg.nbit_wx)

        # cosine range [-1, 1] => Rmax = 1
        scale_wx = float(qmax_wx)  # since Rmax=1

        num_levels = 2 * qmax_w + 1
        # Levels correspond to quantized w in [1..num_levels]
        levels = (np.arange(1, num_levels + 1, dtype=np.float64) / float(scale_w)).reshape(-1, 1)  # [num_levels,1]
        # x: [L] -> [1,L]
        x_row = x.reshape(1, -1).astype(np.float64)

        lut = np.cos(levels * x_row) * scale_wx
        lut_int = np.round(lut).astype(np.int32)  # store as int
        return lut_int, scale_wx, qmax_w

    @classmethod
    def from_float_layer(
        cls,
        layer: nn.Module,
        *,
        qcfg: CosConvQuantConfig,
        w_max: float,
    ) -> Tuple["QuantizedCosConvFloatIO", Dict[str, Any]]:
        """
        Create a quantized float-IO CosConv layer from a trained CosConvLayer.

        Args:
            layer: trained CosConvLayer
            qcfg: bit-width config
            w_max: global max abs(w) used to define w quantization scale (align with original pipeline)

        Returns:
            (quant_layer, meta)
        """
        # Accept both toolbox CosConvLayer and user-defined CosConv-like modules
        if not _is_cosconv_like(layer):
            raise TypeError(
                "layer must be a CosConvLayer (toolbox) or a CosConv-like module with attributes: "
                "A, w, x_buffer, filter_length, num_stride, groups"
            )

        # Pull parameters to CPU numpy
        A = layer.A.detach().cpu().numpy().astype(np.float64)        # [out, in_per_group]
        w = np.abs(layer.w.detach().cpu().numpy().astype(np.float64)) # [out, in_per_group]
        x = layer.x_buffer.detach().cpu().numpy().reshape(-1).astype(np.float64)  # [L]

        # ---- Quantize w (positive integer levels [1..2*Qmax_w+1]) ----
        qmax_w = getQ(qcfg.nbit_w)
        num_levels = 2 * qmax_w + 1

        w_max = float(w_max)
        if not np.isfinite(w_max) or w_max <= qcfg.eps:
            # Degenerate case: all w ~ 0; avoid divide by zero
            w_max = 1.0

        scale_w = float(num_levels / w_max)  # (2*Qmax_w+1)/w_max
        w_q = np.clip(np.round(w * scale_w), 1, num_levels).astype(np.int32)  # [out,in]

        # ---- LUT for cos(w*x) ----
        lut_int, scale_wx, _qmax_w = cls._build_cosine_lut(x=x, qcfg=qcfg, scale_w=scale_w)  # [num_levels,L]

        # Gather LUT rows for each (out,in) w_q
        # lut_int[w_q-1] => shape [out,in,L]
        lut_sel_int = lut_int[w_q - 1]

        # ---- Quantize A ----
        qmax_A = getQ(qcfg.nbit_A)
        A_max = float(np.max(np.abs(A))) if A.size else 0.0
        if not np.isfinite(A_max) or A_max <= qcfg.eps:
            A_max = 1.0
        scale_A = float(qmax_A / A_max)
        A_q = np.round(A * scale_A).astype(np.int32)  # [out,in]

        # ---- Dequantize to float weights for conv1d ----
        # W_float ≈ (A_q/scale_A) * (lut_sel_int/scale_wx)
        W_float = (A_q.astype(np.float64) / scale_A)[:, :, None] * (lut_sel_int.astype(np.float64) / scale_wx)
        W_float = torch.tensor(W_float, dtype=torch.float32)

        # Bias handling (optional)
        bias = None
        if getattr(layer, "Bias", None) is not None:
            b = layer.Bias
            if torch.is_tensor(b):
                bias = b.detach().to(torch.float32).cpu()
            else:
                bias = torch.tensor(b, dtype=torch.float32)

        meta: Dict[str, Any] = {
            "nbit_w": int(qcfg.nbit_w),
            "nbit_wx": int(qcfg.nbit_wx),
            "nbit_A": int(qcfg.nbit_A),
            "w_max": float(w_max),
            "scale_w": float(scale_w),
            "scale_wx": float(scale_wx),
            "scale_A": float(scale_A),
            "qmax_w": int(qmax_w),
            "qmax_A": int(qmax_A),
            "num_levels": int(num_levels),
            "filter_length": int(layer.filter_length),
            "stride": int(layer.num_stride),
            "groups": int(layer.groups),
        }

        qlayer = cls(
            filter_length=layer.filter_length,
            stride=layer.num_stride,
            groups=layer.groups,
            weight=W_float,
            bias=bias,
            meta=meta,
        )
        return qlayer, meta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(
            x,
            self.weight_qfloat,
            bias=None if self.bias is None else getattr(self, self.bias),
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=self.groups,
        )
        return y


def _set_module_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    """
    Replace a submodule by its dotted path name.

    Supports:
    - attributes: "layer1.conv"
    - Sequential/ModuleList indices: "layer1.0.conv"
    """
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def find_cosconv_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, layer) for all CosConv-like modules in `model`."""
    layers: List[Tuple[str, nn.Module]] = []
    for name, m in model.named_modules():
        if _is_cosconv_like(m):
            layers.append((name, m))
    return layers


def quantize_model_cosconv_floatio(
    model: nn.Module,
    *,
    qcfg: CosConvQuantConfig,
    w_max: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quantize all CosConv layers inside an arbitrary model.

    Returns:
        (model_inplace, qmeta)
    """
    cos_layers = find_cosconv_layers(model)
    if not cos_layers:
        if verbose:
            print("[INFO] No CosConvLayer found. Nothing to quantize.")
        return model, {"num_cosconv": 0, "layers": {}}

    if w_max is None:
        # Global max abs(w) across the model
        w_vals = []
        for _name, layer in cos_layers:
            w_vals.append(float(torch.abs(layer.w.detach()).max().cpu().item()))
        w_max = max(w_vals) if w_vals else 1.0

    qmeta: Dict[str, Any] = {
        "type": "cosconv_float_io",
        "w_max": float(w_max),
        "bits": {"w": int(qcfg.nbit_w), "wx": int(qcfg.nbit_wx), "A": int(qcfg.nbit_A)},
        "num_cosconv": len(cos_layers),
        "layers": {},
    }

    if verbose:
        print(f"[INFO] Found {len(cos_layers)} CosConvLayer(s). Global w_max={w_max:.6g}")
        print(f"[INFO] Quant bits: nbit_w={qcfg.nbit_w}, nbit_wx={qcfg.nbit_wx}, nbit_A={qcfg.nbit_A}")

    for name, layer in cos_layers:
        qlayer, meta = QuantizedCosConvFloatIO.from_float_layer(layer, qcfg=qcfg, w_max=float(w_max))
        _set_module_by_name(model, name, qlayer)
        qmeta["layers"][name] = meta
        if verbose:
            print(f"  - Quantized CosConv: {name} (groups={meta['groups']}, k={meta['filter_length']})")

    return model, qmeta
