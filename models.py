"""
backbone + 域分类器 + GRL + 顶层 QCDANN 模型

设计要点：
  - BACKBONE_REGISTRY 注册可选 backbone（qcnn / cnn1d / ...）
    新增只需 register('xxx')(YourBackbone) 即可
  - 所有 backbone 暴露统一接口：forward(x)→logits, get_features(x)→100-dim
  - QCDANNModel 是顶层壳：backbone + (可选) DomainClassifier + GRL
    train.py / deploy.py 都通过它单点构建
"""

import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init


# ──────────────────────────────────────────────────────────
# Backbone Registry（加新 backbone 在这里注册即可）
# ──────────────────────────────────────────────────────────

BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_backbone(name: str):
    def deco(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return deco


def build_backbone(name: str, num_classes: int) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"未知 backbone='{name}'，可选: {list(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](num_classes=num_classes)


# ──────────────────────────────────────────────────────────
# ConvQuadraticOperation: out = conv(x,w_r) * conv(x,w_g) + conv(x²,w_b)
# 初始化使初始等价线性卷积，二次项靠训练逐步学出
# ──────────────────────────────────────────────────────────

class ConvQuadraticOperation(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, padding):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.weight_r = Parameter(torch.empty(out_ch, in_ch, ksize))
        self.weight_g = Parameter(torch.empty(out_ch, in_ch, ksize))
        self.weight_b = Parameter(torch.empty(out_ch, in_ch, ksize))
        self.bias_r   = Parameter(torch.empty(out_ch))
        self.bias_g   = Parameter(torch.empty(out_ch))
        self.bias_b   = Parameter(torch.empty(out_ch))

        nn.init.constant_(self.weight_g, 0)
        nn.init.constant_(self.weight_b, 0)
        nn.init.constant_(self.bias_g, 1)
        nn.init.constant_(self.bias_b, 0)
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
        bound = 1.0 / math.sqrt(fan_in)
        init.uniform_(self.bias_r, -bound, bound)

    def forward(self, x):
        c = dict(stride=self.stride, padding=self.padding)
        return (F.conv1d(x,        self.weight_r, self.bias_r, **c)
              * F.conv1d(x,        self.weight_g, self.bias_g, **c)
              + F.conv1d(x.pow(2), self.weight_b, self.bias_b, **c))


# ──────────────────────────────────────────────────────────
# CNN1D backbone（消融用：去掉二次项）
# ──────────────────────────────────────────────────────────

@register_backbone('cnn1d')
class CNN1D(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        C = nn.Conv1d
        self.cnn = nn.Sequential(
            C(1,  16, 64, 8, 28), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2, 2),
            C(16, 32,  3, 1,  1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2),
            C(32, 64,  3, 1,  1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            C(64, 64,  3, 1,  1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            C(64, 64,  3, 1,  1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            C(64, 64,  3, 1,  0), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            # 自适应到固定长度 3 → 展平维度恒为 64*3=192，与 signal_length 解耦。
            # signal_length=2048 时卷积栈本就输出长度 3，此层为 3→3 恒等（PU 行为不变）；
            # 更大窗长（如 UM=4096 输出长度 7）经此层池化到 3 → fc1 无需改维度。
            nn.AdaptiveMaxPool1d(3),
        )
        self.fc1, self.relu1 = nn.Linear(192, 100), nn.ReLU()  # 192 = 64ch * 3
        self.dp,  self.fc2  = nn.Dropout(0.5), nn.Linear(100, num_classes)

    def get_features(self, x):
        return self.relu1(self.fc1(self.cnn(x).view(x.size(0), -1)))

    def forward(self, x):
        return self.fc2(self.dp(self.get_features(x)))


# ──────────────────────────────────────────────────────────
# QCNN backbone（主方法）
# ──────────────────────────────────────────────────────────

@register_backbone('qcnn')
class QCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        Q = ConvQuadraticOperation
        self.cnn = nn.Sequential(
            Q(1,  16, 64, 8, 28), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2, 2),
            Q(16, 32,  3, 1,  1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2),
            Q(32, 64,  3, 1,  1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            Q(64, 64,  3, 1,  1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            Q(64, 64,  3, 1,  1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            Q(64, 64,  3, 1,  0), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            # 自适应到固定长度 3 → 展平维度恒为 64*3=192，与 signal_length 解耦。
            # signal_length=2048 时卷积栈本就输出长度 3，此层为 3→3 恒等（PU 行为不变）；
            # 更大窗长（如 UM=4096 输出长度 7）经此层池化到 3 → fc1 无需改维度。
            nn.AdaptiveMaxPool1d(3),
        )
        self.fc1, self.relu1 = nn.Linear(192, 100), nn.ReLU()  # 192 = 64ch * 3
        self.dp,  self.fc2  = nn.Dropout(0.5), nn.Linear(100, num_classes)

    def get_features(self, x):
        return self.relu1(self.fc1(self.cnn(x).view(x.size(0), -1)))

    def forward(self, x):
        return self.fc2(self.dp(self.get_features(x)))


# ──────────────────────────────────────────────────────────
# 梯度反转层 + 域分类器
# ──────────────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GRLFunction.apply(x, lambda_)


def progressive_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0) -> float:
    p = epoch / max(total_epochs - 1, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim: int = 100, num_domains: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalDomainClassifier(nn.Module):
    """CDAN domain discriminator on the multilinear feature × prediction map."""
    def __init__(self, feat_dim: int = 100, num_classes: int = 10, num_domains: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * num_classes, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_domains),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────
# 顶层模型壳：backbone + (可选) DomainClassifier
# ──────────────────────────────────────────────────────────

class QCDANNModel(nn.Module):
    """
    train.py 和 deploy.py 唯一调用点。
    forward(x) → cls_logits（推理用）
    forward_da(x, lambda_) → (cls_logits, domain_logits, feat)（训练用）
    """
    def __init__(self, backbone_name: str, num_classes: int,
                 use_dann: bool, num_domains: int = 5, feat_dim: int = 100,
                 use_cdan: bool = False):
        super().__init__()
        self.backbone = build_backbone(backbone_name, num_classes)
        self.use_dann = use_dann
        self.use_cdan = use_cdan
        if use_dann:
            self.domain_clf = DomainClassifier(feat_dim, num_domains)
        else:
            self.domain_clf = None
        if use_cdan:
            self.cdan_clf = ConditionalDomainClassifier(feat_dim, num_classes, num_domains)
        else:
            self.cdan_clf = None

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        return self.backbone.get_features(x)

    def forward_da(self, x, lambda_: float):
        feat = self.backbone.get_features(x)                       # (B, 100)
        cls_logits = self.backbone.fc2(self.backbone.dp(feat))     # (B, num_classes)
        domain_logits = None
        if self.use_dann:
            rev = grad_reverse(feat, lambda_)
            domain_logits = self.domain_clf(rev)                   # (B, num_domains)
        return cls_logits, domain_logits, feat

    def forward_cdan(self, feat: torch.Tensor, cls_logits: torch.Tensor, lambda_: float):
        if not self.use_cdan or self.cdan_clf is None:
            return None
        prob = F.softmax(cls_logits, dim=1)
        conditional = torch.bmm(prob.unsqueeze(2), feat.unsqueeze(1)).flatten(1)
        conditional = conditional / (conditional.norm(dim=1, keepdim=True) + 1e-8)
        return self.cdan_clf(grad_reverse(conditional, lambda_))
