#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CosCNN-DTQ Toolbox
# Copyright (C) 2024-2025 Guoyang Liu
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

"""
Command-line interface for CosCNN-DTQ Toolbox.

This CLI adds two main capabilities:
1) Train (with the SAME cross-validation framework as the GUI toolbox), including custom models.
2) Quantize ONLY CosConv layers inside an arbitrary custom model (float-in / float-out), so you
   can support complex topologies such as ResNet with residual connections.

Examples
--------
Train a custom model:
    python toolbox_cli.py train \
        --model-type custom \
        --model-file path/to/my_model.py \
        --model-class CustomModel \
        --sig-data data/sigData.json \
        --label-data data/labelData.json \
        --num-folds 10 \
        --max-epochs 240 \
        --batch-size 512

Quantize all fold checkpoints produced by training (and evaluate accuracy):
    python toolbox_cli.py quantize \
        --trained-model-dir trained_models \
        --model-type custom \
        --model-file path/to/my_model.py \
        --out-dir quantized_models

Quantize a single checkpoint:
    python toolbox_cli.py quantize \
        --checkpoint trained_models/fold_1/model_fold_1.pth \
        --model-type custom \
        --model-file path/to/my_model.py \
        --output quantized_models/quantized_model_fold_1.pth
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# Ensure imports work no matter where the user runs the CLI from
# ---------------------------------------------------------------------
TOOLBOX_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = TOOLBOX_ROOT / "train"
QUANT_DIR = TOOLBOX_ROOT / "quantization"

import sys
sys.path.insert(0, str(TOOLBOX_ROOT))
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(QUANT_DIR))

from config import TrainingConfig                 # from train/config.py (via sys.path)
from data_manager import DataManager              # from train/data_manager.py
from trainer import ModelTrainer                  # from train/trainer.py
from models.model import CosCNN, GroupCosCNN      # from models/model.py
from quantization.custom_quantizer import CosConvQuantConfig, quantize_model_cosconv_floatio


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # Accept JSON list: "[4,8,16]" or comma-separated: "4,8,16"
    if s.startswith("["):
        return [int(x) for x in json.loads(s)]
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_groups_list(s: Optional[str]) -> Optional[List[Optional[int]]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("["):
        raw = json.loads(s)
    else:
        raw = [x.strip() for x in s.split(",")]
    out: List[Optional[int]] = []
    for v in raw:
        if v is None:
            out.append(None)
        elif isinstance(v, str) and v.lower() in {"none", "null", "auto", ""}:
            out.append(None)
        else:
            out.append(int(v))
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_custom_model_class(model_file: str, model_class: str):
    model_file = os.path.abspath(model_file)
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Custom model file not found: {model_file}")

    spec = importlib.util.spec_from_file_location("user_custom_model", model_file)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    if not hasattr(module, model_class):
        raise AttributeError(f"Model class '{model_class}' not found in {model_file}")
    return getattr(module, model_class)


def _instantiate_custom_model(
    cls,
    *,
    num_classes: int,
    in_channels: Optional[int],
    input_length: Optional[int],
    filter_length: Optional[int],
    num_filters_list: Optional[List[int]],
    groups_list: Optional[List[Optional[int]]],
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Instantiate a user-provided custom model.
    """
    extra_kwargs = extra_kwargs or {}

    sig = inspect.signature(cls)
    kwargs: Dict[str, Any] = dict(extra_kwargs)

    accepts_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    def maybe_set(k: str, v: Any, *, allow_varkw: bool = False):
        if v is None:
            return
        if k in kwargs:
            return
        if k in sig.parameters:
            kwargs[k] = v
        elif allow_varkw and accepts_varkw:
            kwargs[k] = v

    # Essential keys (safe to pass through **kwargs factories)
    maybe_set("num_classes", int(num_classes), allow_varkw=True)
    maybe_set("in_channels", None if in_channels is None else int(in_channels), allow_varkw=True)
    maybe_set("input_length", None if input_length is None else int(input_length), allow_varkw=True)

    # Optional keys (ONLY if explicitly supported)
    maybe_set("filter_length", None if filter_length is None else int(filter_length), allow_varkw=False)
    maybe_set("num_filters_list", num_filters_list, allow_varkw=False)
    maybe_set("groups_list", groups_list, allow_varkw=False)

    # Backward compatibility with the GUI expectation: CustomModel(num_classes=...)
    if "num_classes" not in kwargs and ("num_classes" in sig.parameters or accepts_varkw):
        kwargs["num_classes"] = int(num_classes)

    # Try instantiation with the prepared kwargs.
    try:
        return cls(**kwargs)
    except TypeError as e:
        # Retry with minimal kwargs (GUI-style) to help users who have strict signatures.
        minimal_kwargs: Dict[str, Any] = dict(extra_kwargs)
        if "num_classes" in sig.parameters or accepts_varkw:
            minimal_kwargs["num_classes"] = int(num_classes)
        try:
            return cls(**minimal_kwargs)
        except TypeError:
            raise TypeError(
                f"Failed to instantiate custom model callable '{getattr(cls, '__name__', str(cls))}'.\n"
                f"Signature: {sig}\n"
                f"Tried kwargs (full): {kwargs}\n"
                f"Tried kwargs (minimal): {minimal_kwargs}\n"
                f"Original error: {e}"
            )

def _infer_in_channels_from_model(model: torch.nn.Module) -> Optional[int]:
    # Try to infer from first Conv1d / CosConv layer
    for _name, m in model.named_modules():
        if hasattr(m, "in_channels") and isinstance(getattr(m, "in_channels"), int):
            return int(getattr(m, "in_channels"))
        if hasattr(m, "num_channels") and hasattr(m, "groups"):
            try:
                return int(getattr(m, "num_channels")) * int(getattr(m, "groups"))
            except Exception:
                pass
    return None


def _save_fold_checkpoint(
    *,
    out_dir: Path,
    fold: int,
    model: torch.nn.Module,
    fold_config: Dict[str, Any],
    fold_data: Dict[str, np.ndarray],
):
    fold_dir = out_dir / f"fold_{fold}"
    _ensure_dir(fold_dir)

    ckpt = {"state_dict": model.state_dict(), "config": fold_config}
    torch.save(ckpt, fold_dir / f"model_fold_{fold}.pth")

    with open(fold_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(fold_config, f, indent=4)

    np.savez(
        fold_dir / "data.npz",
        train_data=fold_data["train_data"],
        train_labels=fold_data["train_labels"],
        val_data=fold_data["val_data"],
        val_labels=fold_data["val_labels"],
        test_data=fold_data["test_data"],
        test_labels=fold_data["test_labels"],
    )


def _save_best_model(
    *,
    out_dir: Path,
    best_model_state: Dict[str, Any],
    best_config: Dict[str, Any],
    best_fold_data: Dict[str, np.ndarray],
):
    best_dir = out_dir / "best_model"
    _ensure_dir(best_dir)

    # Keep the same behavior as GUI: best_model/model.pth is a raw state_dict
    torch.save(best_model_state, best_dir / "model.pth")

    with open(best_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4)

    np.savez(
        best_dir / "data.npz",
        train_data=best_fold_data["train_data"],
        train_labels=best_fold_data["train_labels"],
        val_data=best_fold_data["val_data"],
        val_labels=best_fold_data["val_labels"],
        test_data=best_fold_data["test_data"],
        test_labels=best_fold_data["test_labels"],
    )


def _export_fold_data_json(
    data_export_dir: Path,
    fold: int,
    XTrain: torch.Tensor,
    YTrain: torch.Tensor,
    XVal: torch.Tensor,
    YVal: torch.Tensor,
    XTest: torch.Tensor,
    YTest: torch.Tensor,
):
    _ensure_dir(data_export_dir)
    out_file = data_export_dir / f"fold_{fold}_data.json"
    payload = {
        "iFold": int(fold),
        "XTrain": XTrain.cpu().numpy().tolist(),
        "YTrain": YTrain.cpu().numpy().tolist(),
        "XVal": XVal.cpu().numpy().tolist(),
        "YVal": YVal.cpu().numpy().tolist(),
        "XTest": XTest.cpu().numpy().tolist(),
        "YTest": YTest.cpu().numpy().tolist(),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return out_file


# ---------------------------------------------------------------------
# Evaluation Helpers (for Quantization)
# ---------------------------------------------------------------------
def _build_test_loader_from_fold_npz(checkpoint_path: Path, batch_size: int = 512) -> Tuple[Optional[DataLoader], str]:
    """
    Attempts to find 'data.npz' in the same directory as the checkpoint.
    Returns (DataLoader, npz_path_string).
    """
    ckpt_dir = checkpoint_path.parent
    npz_path = ckpt_dir / "data.npz"
    
    if not npz_path.exists():
        # Fallback: maybe we are in a subfolder but data is one level up?
        # But standard structure is fold_x/data.npz
        return None, str(npz_path)

    try:
        npz = np.load(npz_path, allow_pickle=True)
        # Check if test_data exists
        if "test_data" not in npz or "test_labels" not in npz:
            return None, str(npz_path)
            
        x_test = npz["test_data"].astype(np.float32)
        y_test = npz["test_labels"].astype(np.int64)

        x_test_t = torch.from_numpy(x_test)
        y_test_t = torch.from_numpy(y_test)

        ds = TensorDataset(x_test_t, y_test_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        return loader, str(npz_path)
    except Exception as e:
        print(f"[WARN] Failed to load data.npz: {e}")
        return None, str(npz_path)

def _evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Simple evaluation loop to check accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:
        return 0.0
    return float(correct) / total


# ---------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    # Load config (defaults -> optional config.json -> CLI overrides)
    cfg = TrainingConfig()

    if args.config is not None:
        cfg_path = Path(args.config).expanduser().resolve()
        cfg_data = _load_json(cfg_path)
        cfg.update(**cfg_data)

    # CLI overrides
    if args.input_length is not None:
        cfg.input_length = int(args.input_length)
    if args.filter_length is not None:
        cfg.filter_length = int(args.filter_length)
    if args.num_filters_list is not None:
        cfg.num_filters_list = _parse_int_list(args.num_filters_list) or cfg.num_filters_list
    if args.groups_list is not None:
        cfg.groups_list = _parse_groups_list(args.groups_list)
    if args.max_epochs is not None:
        cfg.max_epochs = int(args.max_epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_folds is not None:
        cfg.num_folds = int(args.num_folds)
    if args.model_type is not None:
        cfg.model_type = str(args.model_type)

    cfg.sig_data_path = str(Path(args.sig_data).expanduser().resolve())
    cfg.label_data_path = str(Path(args.label_data).expanduser().resolve())

    cfg.custom_model_path = args.model_file
    # Keep quantization off during CLI training by default
    cfg.perform_quantization = False

    # Directories
    out_dir = Path(args.out_dir).expanduser().resolve()
    data_export_dir = Path(args.data_export_dir).expanduser().resolve()
    _ensure_dir(out_dir)
    _ensure_dir(data_export_dir)

    # Load data
    dm = DataManager(sig_path=None, divide=cfg.num_folds, data_export_dir=str(data_export_dir))
    sigData, labelData, cv_indices, num_classes = dm.load_and_preprocess_data(
        sig_data_path=cfg.sig_data_path,
        label_data_path=cfg.label_data_path,
    )

    # Determine in_channels
    if len(sigData.shape) == 3:
        cfg.in_channels = int(sigData.shape[1])
    elif len(sigData.shape) == 2:
        print("[WARN] Loaded sigData is 2D. Assuming 1 channel and reshaping to (N,1,L).")
        sigData = np.expand_dims(sigData, axis=1)
        cfg.in_channels = 1
    else:
        raise ValueError(f"Unexpected sigData shape: {sigData.shape}")

    # Export in_channels + cv_indices for later quantization
    with open(data_export_dir / "in_channels.json", "w", encoding="utf-8") as f:
        json.dump(int(cfg.in_channels), f)
    with open(data_export_dir / "cv_indices.json", "w", encoding="utf-8") as f:
        json.dump(cv_indices.tolist(), f)

    # Device
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    trainer = ModelTrainer(cfg, device)

    print("=" * 80)
    print("CosCNN-DTQ CLI Training")
    print("=" * 80)
    print(f"Model type: {cfg.model_type}")
    if cfg.model_type == "custom":
        print(f"Custom model file: {cfg.custom_model_path}")
        print(f"Custom model class: {args.model_class}")
    print(f"Input length: {cfg.input_length}")
    print(f"In channels: {cfg.in_channels}")
    print(f"Num classes: {num_classes}")
    print(f"Folds: {cfg.num_folds}")
    print(f"Epochs: {cfg.max_epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Output dir: {out_dir}")
    print(f"Data export dir: {data_export_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    # Cross-validation
    best_fold = 0
    best_accuracy = 0.0
    best_model_state = None
    best_fold_data = None
    accuracies: List[float] = []
    all_confusion = []

    # Optional: extra kwargs for custom model instantiation
    extra_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}

    for fold in range(1, cfg.num_folds + 1):
        print(f"\n== Fold {fold}/{cfg.num_folds} ==")

        itest = (cv_indices == fold)
        itrainval = ~itest

        XTrainVal = sigData[itrainval]
        YTrainVal = labelData[itrainval]

        itrain, ival = DataManager.get_train_val_idx(YTrainVal)
        XTrain = XTrainVal[itrain]
        YTrain = YTrainVal[itrain]
        XVal = XTrainVal[ival]
        YVal = YTrainVal[ival]
        XTest = sigData[itest]
        YTest = labelData[itest]

        # Torch tensors
        XTrain_t = torch.tensor(XTrain, dtype=torch.float32)
        YTrain_t = torch.tensor(YTrain, dtype=torch.long)
        XVal_t = torch.tensor(XVal, dtype=torch.float32)
        YVal_t = torch.tensor(YVal, dtype=torch.long)
        XTest_t = torch.tensor(XTest, dtype=torch.float32)
        YTest_t = torch.tensor(YTest, dtype=torch.long)

        # (Optional) export fold data JSON for quantization calibration (matches GUI behavior)
        if args.export_fold_json:
            out_file = _export_fold_data_json(
                data_export_dir=data_export_dir,
                fold=fold,
                XTrain=XTrain_t, YTrain=YTrain_t,
                XVal=XVal_t, YVal=YVal_t,
                XTest=XTest_t, YTest=YTest_t,
            )
            print(f"[INFO] Exported fold data JSON: {out_file}")

        # Dataloaders
        train_loader = DataLoader(TensorDataset(XTrain_t, YTrain_t), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(XVal_t, YVal_t), batch_size=cfg.batch_size)
        test_loader = DataLoader(TensorDataset(XTest_t, YTest_t), batch_size=cfg.batch_size)

        # Init model
        if cfg.model_type == "default":
            model = CosCNN(
                input_length=cfg.input_length,
                in_channels=cfg.in_channels,
                num_classes=num_classes,
                filter_length=cfg.filter_length,
                num_filters_list=cfg.num_filters_list,
            )
        elif cfg.model_type == "group":
            model = GroupCosCNN(
                input_length=cfg.input_length,
                in_channels=cfg.in_channels,
                num_classes=num_classes,
                filter_length=cfg.filter_length,
                num_filters_list=cfg.num_filters_list,
                groups_list=cfg.groups_list,
            )
        else:
            # custom
            if not cfg.custom_model_path:
                raise ValueError("For model_type=custom, you must provide --model-file")
            cls = _load_custom_model_class(cfg.custom_model_path, args.model_class)
            model = _instantiate_custom_model(
                cls,
                num_classes=num_classes,
                in_channels=cfg.in_channels,
                input_length=cfg.input_length,
                filter_length=cfg.filter_length,
                num_filters_list=cfg.num_filters_list,
                groups_list=cfg.groups_list,
                extra_kwargs=extra_kwargs,
            )

        model.to(device)
        model_trained, test_acc, conf_matrix = trainer.train_model(model, train_loader, val_loader, test_loader)

        print(f"[RESULT] Fold {fold} Test Accuracy: {test_acc*100:.2f}%")
        accuracies.append(float(test_acc))
        all_confusion.append(conf_matrix)

        # Save fold
        fold_config = {
            "input_length": int(cfg.input_length),
            "num_classes": int(num_classes),
            "filter_length": int(cfg.filter_length),
            "num_filters_list": list(cfg.num_filters_list),
            "groups_list": cfg.groups_list,
            "model_type": str(cfg.model_type),
            "fold_number": int(fold),
            "accuracy": float(test_acc),
            "confusion_matrix": conf_matrix.tolist(),
            "date_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if cfg.model_type == "custom":
            fold_config["custom_model_path"] = os.path.abspath(cfg.custom_model_path)
            fold_config["custom_model_class"] = str(args.model_class)
            if extra_kwargs:
                fold_config["custom_model_kwargs"] = extra_kwargs

        fold_data_np = {
            "train_data": XTrain,
            "train_labels": YTrain,
            "val_data": XVal,
            "val_labels": YVal,
            "test_data": XTest,
            "test_labels": YTest,
        }
        _save_fold_checkpoint(out_dir=out_dir, fold=fold, model=model_trained, fold_config=fold_config, fold_data=fold_data_np)

        # Track best fold
        if test_acc >= best_accuracy:
            best_accuracy = float(test_acc)
            best_fold = int(fold)
            best_model_state = model_trained.state_dict()
            best_fold_data = fold_data_np

    # Save best summary
    avg_conf_matrix = np.mean(all_confusion, axis=0) if all_confusion else None
    best_config = {
        "input_length": int(cfg.input_length),
        "num_classes": int(num_classes),
        "filter_length": int(cfg.filter_length),
        "num_filters_list": list(cfg.num_filters_list),
        "groups_list": cfg.groups_list,
        "model_type": str(cfg.model_type),
        "best_fold": int(best_fold),
        "best_accuracy": float(best_accuracy),
        "average_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
        "average_confusion_matrix": avg_conf_matrix.tolist() if avg_conf_matrix is not None else None,
        "date_saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if cfg.model_type == "custom":
        best_config["custom_model_path"] = os.path.abspath(cfg.custom_model_path) if cfg.custom_model_path else None
        best_config["custom_model_class"] = str(args.model_class)
        if extra_kwargs:
            best_config["custom_model_kwargs"] = extra_kwargs

    assert best_model_state is not None and best_fold_data is not None, "No best model found (training may have failed)"
    _save_best_model(out_dir=out_dir, best_model_state=best_model_state, best_config=best_config, best_fold_data=best_fold_data)

    print("\n" + "=" * 80)
    print("✅ Training complete")
    print(f"Best fold: {best_fold}")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Mean accuracy: {np.mean(accuracies)*100:.2f}%")
    print("=" * 80)


# ---------------------------------------------------------------------
# Quantize command
# ---------------------------------------------------------------------
def _load_checkpoint_any(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load checkpoint from either:
      - fold checkpoint: {"state_dict": ..., "config": ...}
      - raw state_dict (best_model/model.pth): {tensor...}
    Returns:
        (state_dict, config)
    """
    obj = torch.load(str(path), weights_only=False, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
        config = obj.get("config", {})
        return state_dict, config

    # Assume raw state_dict
    state_dict = obj
    # try to load config.json in the same directory
    cfg_path = path.parent / "config.json"
    config = _load_json(cfg_path) if cfg_path.exists() else {}
    return state_dict, config


def _find_fold_checkpoints(trained_dir: Path) -> List[Path]:
    """Find fold_i/model_fold_i.pth under trained_dir."""
    out: List[Path] = []
    if not trained_dir.exists():
        return out
    for p in sorted(trained_dir.glob("fold_*")):
        if p.is_dir():
            # Try the canonical file name
            m = list(p.glob("model_fold_*.pth"))
            if m:
                out.append(m[0])
    return out


def cmd_quantize(args: argparse.Namespace) -> None:
    # Resolve inputs
    checkpoints: List[Path] = []
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint).expanduser().resolve()]
    elif args.trained_model_dir:
        checkpoints = _find_fold_checkpoints(Path(args.trained_model_dir).expanduser().resolve())
    else:
        raise ValueError("You must provide either --checkpoint or --trained-model-dir")

    if not checkpoints:
        raise FileNotFoundError("No checkpoints found to quantize.")

    # Quant config
    qcfg = CosConvQuantConfig(
        nbit_w=int(args.nbit_w),
        nbit_wx=int(args.nbit_wx),
        nbit_A=int(args.nbit_a),
    )

    # Output handling
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if args.output:
        out_path_single = Path(args.output).expanduser().resolve()
        _ensure_dir(out_path_single.parent)
    else:
        out_path_single = None
        if out_dir is None:
            out_dir = TOOLBOX_ROOT / "quantized_models"
        _ensure_dir(out_dir)

    # Device for evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Optional model kwargs (for custom model instantiation)
    extra_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}

    # Accumulators for summary
    stats = {
        "float_acc": [],
        "quant_acc": []
    }

    for ckpt_path in checkpoints:
        print("\n" + "=" * 80)
        print(f"Processing: {ckpt_path.name}")
        print("=" * 80)

        state_dict, config = _load_checkpoint_any(ckpt_path)
        model_type = str(args.model_type or config.get("model_type", "custom"))

        # Determine in_channels
        in_channels = args.in_channels
        if in_channels is None:
            # Try to read from data/in_channels.json if user provides data-export-dir
            if args.data_export_dir:
                p = Path(args.data_export_dir).expanduser().resolve() / "in_channels.json"
                if p.exists():
                    in_channels = int(_load_json(p))
        in_channels = int(in_channels) if in_channels is not None else None

        input_length = config.get("input_length", args.input_length)
        num_classes = config.get("num_classes", args.num_classes)
        filter_length = config.get("filter_length", args.filter_length)
        num_filters_list = config.get("num_filters_list", None)
        groups_list = config.get("groups_list", None)

        # Instantiate model
        if model_type == "default":
            if num_classes is None:
                raise ValueError("num_classes missing. Provide --num-classes or use a fold checkpoint with config.")
            if in_channels is None:
                in_channels = 1
            model = CosCNN(
                input_length=int(input_length),
                in_channels=int(in_channels),
                num_classes=int(num_classes),
                filter_length=int(filter_length),
                num_filters_list=list(num_filters_list),
            )
        elif model_type == "group":
            if num_classes is None:
                raise ValueError("num_classes missing. Provide --num-classes or use a fold checkpoint with config.")
            if in_channels is None:
                in_channels = 1
            model = GroupCosCNN(
                input_length=int(input_length),
                in_channels=int(in_channels),
                num_classes=int(num_classes),
                filter_length=int(filter_length),
                num_filters_list=list(num_filters_list),
                groups_list=groups_list,
            )
        else:
            # custom model
            model_file = args.model_file or config.get("custom_model_path", None)
            model_class = args.model_class or config.get("custom_model_class", "CustomModel")
            if not model_file:
                raise ValueError("For model_type=custom, you must provide --model-file (or it must exist in checkpoint config).")
            if num_classes is None:
                raise ValueError("num_classes missing. Provide --num-classes or use a fold checkpoint with config.")

            cls = _load_custom_model_class(model_file, model_class)
            model = _instantiate_custom_model(
                cls,
                num_classes=int(num_classes),
                in_channels=in_channels,
                input_length=None if input_length is None else int(input_length),
                filter_length=None if filter_length is None else int(filter_length),
                num_filters_list=num_filters_list if isinstance(num_filters_list, list) else None,
                groups_list=groups_list if isinstance(groups_list, list) else None,
                extra_kwargs=extra_kwargs or config.get("custom_model_kwargs", {}),
            )
            if in_channels is None:
                inferred = _infer_in_channels_from_model(model)
                if inferred is not None:
                    in_channels = inferred

        # Load float weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading state_dict (strict=False): {missing[:8]}{'...' if len(missing)>8 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading state_dict (strict=False): {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

        model.eval()
        
        # ---------------------------------------------------------------------
        # Pre-Quantization Evaluation (Float)
        # ---------------------------------------------------------------------
        test_loader = None
        if not args.no_eval:
            test_loader, npz_path = _build_test_loader_from_fold_npz(ckpt_path, batch_size=args.eval_batch_size)
            if test_loader is not None:
                model.to(device)
                float_acc = _evaluate_accuracy(model, test_loader, device)
                print(f"[EVAL] Float Accuracy:     {float_acc * 100:.2f}%")
                stats["float_acc"].append(float_acc)
            else:
                print(f"[WARN] Test data not found at {npz_path}, skipping evaluation.")

        # ---------------------------------------------------------------------
        # Quantize CosConv layers only
        # ---------------------------------------------------------------------
        model_q, qmeta = quantize_model_cosconv_floatio(model, qcfg=qcfg, verbose=True)

        # ---------------------------------------------------------------------
        # Post-Quantization Evaluation
        # ---------------------------------------------------------------------
        if test_loader is not None:
            model_q.to(device)
            quant_acc = _evaluate_accuracy(model_q, test_loader, device)
            print(f"[EVAL] Quantized Accuracy: {quant_acc * 100:.2f}%")
            stats["quant_acc"].append(quant_acc)

        # Prepare output path
        if out_path_single is not None:
            out_path = out_path_single
        else:
            # If ckpt looks like fold_x/model_fold_x.pth, preserve fold index
            fold_name = ckpt_path.parent.name
            out_path = out_dir / f"{fold_name}_quantized.pth" if fold_name.startswith("fold_") else out_dir / (ckpt_path.stem + "_quantized.pth")

        # Save checkpoint
        out_ckpt = {
            "state_dict": model_q.state_dict(),
            "config": {
                **(config or {}),
                "model_type": model_type,
                "quantization": qmeta,
                "date_quantized": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        # For custom, preserve model file/class metadata
        if model_type == "custom":
            out_ckpt["config"]["custom_model_path"] = os.path.abspath(args.model_file or config.get("custom_model_path", ""))
            out_ckpt["config"]["custom_model_class"] = str(args.model_class or config.get("custom_model_class", "CustomModel"))
            if extra_kwargs:
                out_ckpt["config"]["custom_model_kwargs"] = extra_kwargs

        torch.save(out_ckpt, out_path)
        print(f"[OK] Saved quantized checkpoint: {out_path}")

    # Summary if multiple folds
    if len(checkpoints) > 1 and stats["quant_acc"]:
        print("\n" + "=" * 80)
        print("Quantization Summary")
        print("=" * 80)
        if stats["float_acc"]:
            print(f"Mean Float Acc:     {np.mean(stats['float_acc']) * 100:.2f}%")
        print(f"Mean Quantized Acc: {np.mean(stats['quant_acc']) * 100:.2f}%")
        print("=" * 80)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="toolbox_cli.py", description="CosCNN-DTQ Toolbox CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Train a model with the same CV framework as the GUI.")
    p_train.add_argument("--config", type=str, default=None, help="Path to a training_config.json (optional).")
    p_train.add_argument("--sig-data", type=str, required=True, help="Path to sigData.json")
    p_train.add_argument("--label-data", type=str, required=True, help="Path to labelData.json")
    p_train.add_argument("--model-type", type=str, default="default", choices=["default", "group", "custom"], help="Model type.")
    p_train.add_argument("--model-file", type=str, default=None, help="Custom model .py path (required if model-type=custom).")
    p_train.add_argument("--model-class", type=str, default="CustomModel", help="Custom model class name inside .py file.")
    p_train.add_argument("--model-kwargs", type=str, default=None, help="Extra kwargs JSON string for custom model init.")
    p_train.add_argument("--input-length", type=int, default=None)
    p_train.add_argument("--filter-length", type=int, default=None)
    p_train.add_argument("--num-filters-list", type=str, default=None, help="e.g. '4,8,16,32,64' or '[4,8,16]'.")
    p_train.add_argument("--groups-list", type=str, default=None, help="Per-layer groups list for GroupCosCNN, e.g. '[2,2,4]' or 'auto,auto,2'.")
    p_train.add_argument("--max-epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--num-folds", type=int, default=None)
    p_train.add_argument("--device", type=str, default=None, help="e.g. 'cpu' or 'cuda:0'.")
    p_train.add_argument("--out-dir", type=str, default="trained_models", help="Output directory for checkpoints (default: trained_models).")
    p_train.add_argument("--data-export-dir", type=str, default="data", help="Directory to export fold splits (default: data).")
    p_train.add_argument("--export-fold-json", dest="export_fold_json", action="store_true", default=True,
                         help="Export fold_{i}_data.json to data-export-dir (default: enabled).")
    p_train.add_argument("--no-export-fold-json", dest="export_fold_json", action="store_false",
                         help="Disable exporting fold_{i}_data.json (can save disk space).")
    p_train.set_defaults(func=cmd_train)

    # ---- quantize ----
    p_q = sub.add_parser("quantize", help="Quantize CosConv layers only (float-in / float-out).")
    g_in = p_q.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint (.pth) to quantize.")
    g_in.add_argument("--trained-model-dir", type=str, default=None, help="Directory containing fold_* subfolders to quantize all folds.")
    p_q.add_argument("--model-type", type=str, default="custom", choices=["default", "group", "custom"], help="Model type (default: custom).")
    p_q.add_argument("--model-file", type=str, default=None, help="Custom model .py path (required if model-type=custom unless checkpoint config already has it).")
    p_q.add_argument("--model-class", type=str, default=None, help="Custom model class name inside .py file.")
    p_q.add_argument("--model-kwargs", type=str, default=None, help="Extra kwargs JSON string for custom model init.")
    p_q.add_argument("--data-export-dir", type=str, default=None, help="If provided, will try to read in_channels.json from here.")
    p_q.add_argument("--in-channels", type=int, default=None, help="Override input channels for model init.")
    p_q.add_argument("--input-length", type=int, default=None, help="Override input_length if checkpoint config missing.")
    p_q.add_argument("--num-classes", type=int, default=None, help="Override num_classes if checkpoint config missing.")
    p_q.add_argument("--filter-length", type=int, default=None, help="Override filter_length if checkpoint config missing.")
    p_q.add_argument("--nbit-w", type=int, default=12, help="Bit width for w quantization (default: 12).")
    p_q.add_argument("--nbit-wx", type=int, default=8, help="Bit width for cosine LUT values (default: 8).")
    p_q.add_argument("--nbit-a", type=int, default=8, help="Bit width for A quantization (default: 8).")
    p_q.add_argument("--out-dir", type=str, default="quantized_models", help="Output directory (used when quantizing many checkpoints).")
    p_q.add_argument("--output", type=str, default=None, help="Output file path (used when quantizing a single checkpoint).")
    # New evaluation args
    p_q.add_argument("--eval-batch-size", type=int, default=512, help="Batch size for accuracy evaluation (default: 512).")
    p_q.add_argument("--no-eval", action="store_true", help="Skip accuracy evaluation.")
    p_q.set_defaults(func=cmd_quantize)

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()