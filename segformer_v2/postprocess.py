# -*- coding: utf-8 -*-
import math

import numpy as np
import torch

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def validate_confidence_threshold(value):
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f'confidence_threshold 必须在 [0, 1] 内，实际为 {value}')
    return threshold


def resolve_confidence_threshold(cfg, override=None):
    value = override if override is not None else cfg.get(
        'confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
    return validate_confidence_threshold(value)


def logits_to_labels(logits, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """将 (C,H,W) 融合 logits 转为标签，低置信度像素归为背景 0。"""
    threshold = validate_confidence_threshold(confidence_threshold)
    if logits.ndim < 1 or logits.shape[0] < 1:
        raise ValueError('logits 的类别维度不能为空')
    max_logits = np.max(logits, axis=0)
    exp_shifted = np.exp(logits - max_logits[None])
    confidence = 1.0 / np.sum(exp_shifted, axis=0)
    labels = np.argmax(logits, axis=0).astype(np.uint8)
    labels[~(confidence >= threshold)] = 0
    return labels


def torch_logits_to_labels(logits, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """将 (B,C,...) logits 转为标签，低置信度像素归为背景 0。"""
    threshold = validate_confidence_threshold(confidence_threshold)
    if logits.ndim < 2 or logits.shape[1] < 1:
        raise ValueError('logits 的类别维度不能为空')
    work = logits.float()
    max_logits, labels = work.max(dim=1)
    confidence = torch.exp(max_logits - torch.logsumexp(work, dim=1))
    return torch.where(confidence >= threshold, labels, torch.zeros_like(labels))
