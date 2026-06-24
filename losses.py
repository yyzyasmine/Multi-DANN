"""
辅助损失注册表（除分类 + 域对抗外的所有附加项）

LOSS_REGISTRY 里每个条目：
  name → callable(feat_src, feat_tgt) → scalar tensor
  （只接收源域 + 目标域特征，签名统一以便 trainer.py 通用化调用）

加新损失只要 @register_loss('name')，并在 config.py ABLATION_PRESETS
的 aux_losses 里挂上 (name, lambda_attr) 就行，trainer.py 自动循环求和。
"""

from typing import Callable, Dict

import torch


LOSS_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
JOINT_LOSS_REGISTRY: Dict[
    str,
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
] = {}


def register_loss(name: str):
    def deco(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


def get_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise KeyError(f"未知 loss='{name}'，可选: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]


def register_joint_loss(name: str):
    """注册需要同时访问特征与分类预测的联合分布对齐损失。"""
    def deco(fn):
        JOINT_LOSS_REGISTRY[name] = fn
        return fn
    return deco


def get_joint_loss(name: str):
    if name not in JOINT_LOSS_REGISTRY:
        raise KeyError(f"未知 joint loss='{name}'，可选: {list(JOINT_LOSS_REGISTRY)}")
    return JOINT_LOSS_REGISTRY[name]


# ──────────────────────────────────────────────────────────
# CORAL —— 协方差对齐
# ──────────────────────────────────────────────────────────

@register_loss('coral')
def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    s_centered = source - source.mean(0, keepdim=True)
    t_centered = target - target.mean(0, keepdim=True)
    cs = s_centered.t() @ s_centered / (ns - 1 + 1e-8)
    ct = t_centered.t() @ t_centered / (nt - 1 + 1e-8)
    return (cs - ct).pow(2).sum() / (4.0 * d * d)


# ──────────────────────────────────────────────────────────
# Sparse Filtering —— L2-normed L1 稀疏约束
# ──────────────────────────────────────────────────────────

@register_loss('sparse')
def sparse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对源 + 目标特征同时施加稀疏；签名保持 (src, tgt) 以与其他 loss 一致。"""
    feat = torch.cat([source, target], dim=0)
    feat_norm = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
    return feat_norm.abs().mean()


# ──────────────────────────────────────────────────────────
# MMD —— 多核最大均值差异
# ──────────────────────────────────────────────────────────

def _rbf(X: torch.Tensor, Y: torch.Tensor, bw: float) -> torch.Tensor:
    XX = (X * X).sum(dim=1, keepdim=True)
    YY = (Y * Y).sum(dim=1, keepdim=True)
    dist = XX + YY.t() - 2.0 * (X @ Y.t())
    return torch.exp(-dist / (2.0 * bw))


@register_loss('mmd')
def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        comb = torch.cat([source, target], dim=0)
        d2 = torch.cdist(comb, comb).pow(2)
        med = d2[d2 > 0].median().item() if (d2 > 0).any() else 1.0
    bws = [med / 9, med / 3, med, med * 3, med * 9]
    loss = torch.tensor(0.0, device=source.device)
    for bw in bws:
        loss = loss + (_rbf(source, source, bw).mean()
                       - 2.0 * _rbf(source, target, bw).mean()
                       + _rbf(target, target, bw).mean())
    return loss / len(bws)


# ──────────────────────────────────────────────────────────
# JAN —— feature × class-probability 的联合分布 MMD
# ──────────────────────────────────────────────────────────

def _joint_feature(feat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    joint = torch.bmm(prob.unsqueeze(2), feat.unsqueeze(1)).flatten(1)
    return joint / (joint.norm(dim=1, keepdim=True) + 1e-8)


@register_joint_loss('jan')
def jan_loss(source_feat: torch.Tensor,
             target_feat: torch.Tensor,
             source_logits: torch.Tensor,
             target_logits: torch.Tensor) -> torch.Tensor:
    """JAN-style joint distribution alignment using feature and prediction layers."""
    return mmd_loss(_joint_feature(source_feat, source_logits),
                    _joint_feature(target_feat, target_logits))
