"""
Muti-QCDANN 统一配置中心
========================

设计目标：
  - 所有超参数集中在此，train.py / deploy.py 不再硬编码任何数字
  - 通过 `mode` 名称索引 ABLATION_PRESETS，一个名字 → 一组开关 → 训练/部署都从此取
  - 加新模块只需：① 在 losses.py / models.py 里注册名字 ② 在 ABLATION_PRESETS 里引用

用法：
  from config import Config, get_preset
  cfg = Config(mode='E1_QMDCS', dataset='PU')
  preset = get_preset(cfg.mode)        # 取出开关 dict
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import subprocess


def get_git_branch() -> str:
    """获取当前 git 分支名称，失败时返回 'unknown'。"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            # 将分支名中的特殊字符转换为下划线
            branch = branch.replace('/', '_').replace(' ', '_')
            return branch
    except Exception:
        pass
    return 'unknown'


# ──────────────────────────────────────────────────────────
# 数据集 / SNR 域
# ──────────────────────────────────────────────────────────

# 数据集：PU（10 类，64kHz，子目录布局）/ CWRU（10 类，12kHz，平铺布局）。
# 两者独立训练（各自模型），不要求同采样率；终端用 --dataset 切换。
# CWRU 读取参考 E:\Python\QC-DANN：每个 .mat 取含 'DE' 的通道键。
FLAT_DATASETS = ('CWRU',)   # 平铺布局家族（flat_filename_class，块洗牌切分）

# 每个数据集独立的根目录（直接指向类别子目录/文件所在层）。改这里就能切换数据源。
DATA_ROOTS: Dict[str, str] = {
    'PU':   r'E:\Python\PU',
    'CWRU': r'E:\Python\CWRU',
}

# .mat 文件里振动信号所在的键
#   PU   : MATLAB 结构体里 Y[*].Name == 'vibration_1' 的 Data 通道
#   CWRU : 含 'DE'（Drive-End）的键，如 'X118_DE_time'，模糊匹配 'DE' in key
SIGNAL_KEYS: Dict[str, str] = {
    'PU':   'vibration_1',
    'CWRU': 'DE',
}

# 采样率（Hz）—— 仅用于日志/对照；网络只看每窗 signal_length 个样本，
# 各数据集独立训练时不需要强行统一（PU 保持 64kHz 原生，不下采样）。
SAMPLING_RATES: Dict[str, int] = {
    'PU':   64000,
    'CWRU': 12000,
}

# 每数据集独立的滑窗窗长（采样点数）。只改一个数据集，另一个不受影响。
#   PU   : 2048（已很好，93.8%）。
#   CWRU : 2048（同参考项目；CWRU 每类 ~12 万点，小数据靠块内重叠凑样本）。
# Config.signal_length 留空(None)时由此表按 dataset 自动取值；显式传值则覆盖。
SIGNAL_LENGTH_BY_DATASET: Dict[str, int] = {
    'PU':   2048,
    'CWRU': 2048,
}

# 数据布局决定切分策略：
#   multi_file_per_class : 子目录 = 类；每类多个 .mat；按文件个数比例切 train/val/test (PU)
#   flat_filename_class  : .mat 平铺在根目录；类别从文件名解析；每文件块洗牌切 + buffer (CWRU)
DATASET_LAYOUTS: Dict[str, str] = {
    'PU':   'multi_file_per_class',
    'CWRU': 'flat_filename_class',
}

# ── 平铺布局（CWRU）的块洗牌 / 滑窗参数（按数据集；PU 走文件级切分不受此影响）──
# CWRU 数据小：块数需小（否则 block_len<窗长），训练窗靠高重叠凑足目标窗数。
FLAT_NUM_BLOCKS: Dict[str, int] = {'CWRU': 10}        # 每条录音切几块（块洗牌分 train/val/test）
FLAT_SPLIT_BUFFER_FACTOR: Dict[str, float] = {'CWRU': 0.5}  # 块间隔离带 = signal_length * 该值

# 每类目标样本（窗）数：按数据集。本分支做 CWRU/PU 对照，PU 也对齐 CWRU 的
# 750/250/250，避免两个数据集因为样本窗数量不同引入额外变量。
NUM_PER_CLASS_BY_DATASET: Dict[str, Tuple[int, int, int]] = {  # (train, val, test)
    'PU':   (750,  250, 250),
    'CWRU': (750,  250, 250),
}

# 滑窗步长（采样点）：按数据集。值为 None 时 __post_init__ 自动算（PU 用重叠比例/不重叠）。
# 本分支要求 PU 与 CWRU 滑窗重叠率一致：
#   train step=72  → 约 96.5% 重叠（2048 点窗）
#   eval  step=256 → 87.5% 重叠（2048 点窗）
# PU 仍按文件级切分，CWRU 仍按块洗牌切分；这里只对齐滑窗与样本窗数量。
ENC_STEP_TRAIN_BY_DATASET: Dict[str, Optional[int]] = {'PU': 72, 'CWRU': 72}
ENC_STEP_EVAL_BY_DATASET:  Dict[str, Optional[int]] = {'PU': 256, 'CWRU': 256}
TRAIN_WINDOW_OVERLAP: float = 1.0 - 72 / 2048   # 仅作为未显式设置步长的数据集兜底

# PU 类别集合：两套白名单，由 cfg.pu_class_set 选择。默认 '10class'（与 CWRU 10 类对照）。
#
# ── '10class' (默认)：健康 1 + 外圈 3 + 滚动体 3 + 内圈 3 = 10 ──
#   与 CWRU 的 10 类（normal + B/IR/OR 各 3 尺寸）做对照实验。
#
# ── '4class'：纯故障类型 4 分类，每型选一个代表编号（K001/KA01/KB23/KI01）──
#   旧对照设定，保留备用。
#
# 切换方式：把 cfg.pu_class_set 改成 '4class' 即可，不需要改任何代码。
PU_CLASSES_4: List[str] = [
    'K001',     # 健康
    'KA01',     # 外圈   (outer race)
    'KB23',     # 滚动体 (ball bearing)
    'KI01',     # 内圈   (inner race)
]

PU_CLASSES_10: List[str] = [
    'K001',                     # 健康 1
    'KA01', 'KA03', 'KA04',     # 外圈 3
    'KB23', 'KB24', 'KB27',     # 滚动体 3
    'KI01', 'KI03', 'KI04',     # 内圈 3
]

PU_CLASS_SETS: Dict[str, List[str]] = {
    '4class':  PU_CLASSES_4,
    '10class': PU_CLASSES_10,
}

# 旧代码兼容：默认 = 4class（最新主推设定）。
PU_CLASSES: List[str] = PU_CLASSES_4

# CWRU 10 类白名单（E:\Python\CWRU 下 10 个 .mat，每文件 = 一类，12kHz Drive-End）。
# 列表顺序 = 整数标签顺序：健康 0，滚动体 B 1-3，内圈 IR 4-6，外圈 OR 7-9（各 3 个尺寸 007/014/021）。
# 匹配规则：文件名以 "<class>_" 或 "<class>." 开头即归为该类（此处用整文件名 stem，
#   故 'stem.mat'.startswith('stem'+'.') 命中各自文件；大小写敏感）。
CWRU_CLASSES: List[str] = [
    'normal_0_97',                                                  # 健康
    '12k_Drive_End_B007_0_118',  '12k_Drive_End_B014_0_185',  '12k_Drive_End_B021_0_222',   # 滚动体 ×3
    '12k_Drive_End_IR007_0_105', '12k_Drive_End_IR014_0_169', '12k_Drive_End_IR021_0_209',  # 内圈 ×3
    '12k_Drive_End_OR007@6_0_130', '12k_Drive_End_OR014@6_0_197', '12k_Drive_End_OR021@6_0_234',  # 外圈 ×3
]

# 5 个 SNR 域：+6 / +3 / 0 / -3 / -6。已恢复 -6dB（曾因"噪声主导、难学"被移除，
# 现重新纳入以覆盖更低 SNR、检验模型在强噪声下的域适应能力）。仅 clean 仍移除：
#   - clean: 与所有 SNR 域信号分布差异过大，DA 对齐不稳。
# 源域 = +6dB（最易域）：经典 clean→noisy DA 设定——让模型先在干净信号上学到清晰
# 的故障特征，再用 DANN/CORAL/Sparse 把 0dB / -3dB / -6dB 的特征拉过来。多域监督下
# 其它 4 个域同样带标签训练，源域只是"主战场"。
# 想改源域：把 SOURCE_SNR 改成 6 / 3 / 0 / -3 / -6 任一即可，其余代码自动适配。
SNR_DOMAINS: Dict[int, Any] = {0: 6, 1: 3, 2: 0, 3: -3, 4: -6}
SNR_LIST: List[Any] = [6, 3, 0, -3, -6]
SOURCE_SNR: Any = 6


def format_snr(snr) -> str:
    """显示用格式化：数值 → '+6dB' / '-3dB'。"""
    if snr is None:
        return 'clean'
    return f"{snr:+d}dB"


def get_cwru_classes(cfg) -> List[str]:
    """CWRU 固定 10 类（每文件一类，标签 = CWRU_CLASSES 列表顺序）。"""
    return list(CWRU_CLASSES)


def get_pu_classes(cfg) -> List[str]:
    """根据 cfg.pu_class_set 选 PU 白名单（'4class' / '10class'）。"""
    name = getattr(cfg, 'pu_class_set', '4class')
    if name not in PU_CLASS_SETS:
        raise KeyError(f"未知 pu_class_set='{name}'。可选：{list(PU_CLASS_SETS)}")
    return PU_CLASS_SETS[name]


def get_class_names(cfg) -> List[str]:
    """统一入口：根据 dataset 返回当前类名列表（PU 走 pu_class_set；CWRU 固定 10 类）。"""
    if cfg.dataset == 'PU':
        return list(get_pu_classes(cfg))
    if cfg.dataset in FLAT_DATASETS:
        return list(get_cwru_classes(cfg))
    raise ValueError(f"未知 dataset='{cfg.dataset}'")


# ──────────────────────────────────────────────────────────
# 主 Config（所有超参）
# ──────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── 运行身份 ──
    mode: str = 'E1_QMDCS'              # 索引 ABLATION_PRESETS 的名字
    dataset: str = 'PU'                 # 'PU'(10类,64kHz) / 'CWRU'(10类,12kHz)
    seed: int = 42

    # ── 数据划分（先切分再滑窗，避免泄露）──
    # PU: 按文件个数比例切；CWRU(flat): 块洗牌按比例切并加 split_buffer 隔离带
    train_ratio: float = 0.60
    val_ratio:   float = 0.20
    test_ratio:  float = 0.20

    # 平铺布局（CWRU）块洗牌切分：把每条录音切成 flat_num_blocks 等长块，shuffle 后按比例
    # 分到三集；块间留 buffer = signal_length * split_buffer_factor 作隔离带 → 杜绝跨块
    # 时间相邻泄露，且块洗牌让三集在时序上交错分布。留 None → __post_init__ 按 dataset 取
    # （CWRU: 10 块 / buffer 系数 0.5；PU 文件级切分不用这俩）。
    split_buffer_factor: Optional[float] = None
    flat_num_blocks:     Optional[int]   = None

    # 滑窗（留 None → __post_init__ 按 dataset 自动解析；显式传值则覆盖）
    #   signal_length : None → SIGNAL_LENGTH_BY_DATASET[dataset]（PU/CWRU 均 2048）
    #   enc_step_train: None → CWRU 用 ENC_STEP_TRAIN_BY_DATASET；PU 用 窗长*(1-重叠)
    #   enc_step_val/test: None → CWRU 用 ENC_STEP_EVAL_BY_DATASET；PU 用 窗长(不重叠)
    signal_length: Optional[int] = None
    enc_step_train: Optional[int] = None
    enc_step_val:   Optional[int] = None
    enc_step_test:  Optional[int] = None

    # 每类目标样本（窗）数：留 None → __post_init__ 按 dataset 取 NUM_PER_CLASS_BY_DATASET
    # （PU 1500/500/500；CWRU 750/250/250，小数据降重叠）。按各自子段长度自动截断。
    num_train_per_class: Optional[int] = None
    num_val_per_class:   Optional[int] = None
    num_test_per_class:  Optional[int] = None

    # ── PU 类别集合切换（仅 PU 数据集生效，CWRU 忽略）──
    # '10class'（默认）: 健康1+外圈3+滚动体3+内圈3 = 10 类（与 CWRU 10 类对照）
    # '4class' : 每故障类型选一个代表编号 = 4 类（旧对照设定）
    # 影响：PU 数据的 classes 列表、num_classes、缓存 hash（自动落到不同 hash 目录）
    pu_class_set: str = '10class'

    # ── 训练模式 ──
    # 'multi'  : 多域训练 —— 源域 + 5 个目标域均带标签训练（默认）
    # 'single' : 单域训练 —— 只在 single_domain_snr 一个域上训练，其他 5 域仅用于 val/test
    # 单域模式下 progressive_training 失去意义（无目标域可加），会被自动忽略
    training_mode: str = 'multi'
    # 单域模式下使用哪个 SNR；None = clean
    single_domain_snr: Any = None

    # ── 渐进训练 ──
    # True 时按 SNR_LIST 顺序逐 stage 加入目标域：
    #   stage 0  仅源域；
    #   stage k  加入 SNR_LIST 里第 k 个非源域（SNR 从高到低）
    # cfg.epochs 自动均分到 (1 + 非源域数) 个 stage（默认 6 stages）。
    # 模型架构（DANN 域分类器输出维度 = len(SNR_LIST)）保持不变，
    # val/test 仍在全 6 域上跑，方便比较各 stage 的全局进展。
    progressive_training: bool = False

    # ── 训练 ──
    epochs: int = 70
    batch_size: int = 64
    target_batch_size: int = 64
    lr: float = 0.01 #之前都是0.01 测试um数据集使用0.001
    alpha: float = 0.1                  # QCNN 二次项参数组 lr 缩放
    momentum: float = 0.9
    weight_decay: float = 5e-4
    max_grad_norm: float = 1.0

    # ── 早停 ──
    # 平均 ValAcc 连续 patience 个 epoch 无新高则提前终止；曲线和 ckpt 已保留
    patience: int = 20

    # ── 域适应权重 ──
    lambda_max:    float = 1.0          # GRL 调度上限
    lambda_coral:  float = 0.005
    lambda_sparse: float = 0.005
    lambda_mmd:    float = 0.5

    # ── 输出 ──
    results_root: str = 'results'
    no_cuda: bool = False

    # ── 加噪数据磁盘缓存（PU / CWRU 通用）──
    # 第一次生成后写到 cache_dir/<dataset>/<hyper-hash>/<snr>.npz，下次命中直接读盘。
    # 影响输出的超参（classes/ratios/signal_length/步长/num_*/seed，CWRU 额外 split_buffer 与
    # flat_num_blocks）任一项变化 → 自动落到新 hash 目录，不会用错旧缓存。
    # 注意：原始 .mat 文件被换掉/覆盖时不会自动失效 —— 这种情况下手动删 cache_dir/<dataset>/。
    use_cache: bool = True
    cache_dir: str = 'data/cache'

    def __post_init__(self):
        """按 dataset 解析数据参数（仅当字段留 None 时）；显式传入值保持不变。
        PU 与 CWRU 各取各的滑窗/块/样本数，互不影响。"""
        ds = self.dataset
        if self.signal_length is None:
            self.signal_length = SIGNAL_LENGTH_BY_DATASET.get(ds, 2048)
        # 训练步长：CWRU 用专用步长（高重叠）；否则按重叠比例算
        if self.enc_step_train is None:
            v = ENC_STEP_TRAIN_BY_DATASET.get(ds)
            self.enc_step_train = v if v else int(round(self.signal_length * (1.0 - TRAIN_WINDOW_OVERLAP)))
        # val/test 步长：CWRU 用专用步长（小数据需重叠）；否则不重叠=窗长
        eval_step = ENC_STEP_EVAL_BY_DATASET.get(ds)
        if self.enc_step_val is None:
            self.enc_step_val = eval_step if eval_step else self.signal_length
        if self.enc_step_test is None:
            self.enc_step_test = eval_step if eval_step else self.signal_length
        # 每类样本数
        n_tr, n_va, n_te = NUM_PER_CLASS_BY_DATASET.get(ds, (1500, 500, 500))
        if self.num_train_per_class is None: self.num_train_per_class = n_tr
        if self.num_val_per_class   is None: self.num_val_per_class   = n_va
        if self.num_test_per_class  is None: self.num_test_per_class  = n_te
        # 平铺布局块洗牌参数（PU 文件级切分用不到，给个无害默认）
        if self.flat_num_blocks is None:
            self.flat_num_blocks = FLAT_NUM_BLOCKS.get(ds, 100)
        if self.split_buffer_factor is None:
            self.split_buffer_factor = FLAT_SPLIT_BUFFER_FACTOR.get(ds, 1.0)

    # ── 一些便捷属性 ──
    @property
    def num_domains(self) -> int:
        return len(SNR_DOMAINS)

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATES.get(self.dataset, 0)

    def run_dir(self) -> str:
        branch = get_git_branch()
        return os.path.join(self.results_root, f'{branch}_{self.dataset}_{self.mode}_seed{self.seed}')

    def ckpt_path(self) -> str:
        return os.path.join(self.run_dir(), 'checkpoint.pth')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────
# ABLATION_PRESETS：单一字典驱动所有消融
# ──────────────────────────────────────────────────────────
# 字段说明：
#   modules            : 当前组合的模块字母列表（Q/M/D/C/S）
#   backbone           : 'qcnn' | 'cnn1d'（含 Q → qcnn；不含 Q → cnn1d 承载对照）
#   use_target_domains : 是否在训练中迭代目标域 batch（M/D/C/S 任一存在即 True）
#   use_multidomain    : 是否使用目标域分类标签监督（即 M 模块）
#   use_dann           : 是否挂域分类器 + GRL（即 D 模块）
#   aux_losses         : list of (loss_name, lambda_attr)，C/S 分别对应 coral/sparse
#                        lambda_attr 是 Config 上的字段名，运行时读 cfg.<lambda_attr>
#   description        : 一行说明
#
# 添加新消融只需在这里写一行，无需改 train.py / deploy.py / trainer.py。

ABLATION_COMBOS: List[Tuple[str, Tuple[str, ...]]] = [
    ('A0_BASE',   ()),
    ('A1_Q',      ('Q',)),
    ('A2_M',      ('M',)),
    ('A3_D',      ('D',)),
    ('A4_C',      ('C',)),
    ('A5_S',      ('S',)),
    ('B1_QM',     ('Q', 'M')),
    ('B2_QD',     ('Q', 'D')),
    ('B3_QC',     ('Q', 'C')),
    ('B4_QS',     ('Q', 'S')),
    ('B5_MD',     ('M', 'D')),
    ('B6_MC',     ('M', 'C')),
    ('B7_MS',     ('M', 'S')),
    ('B8_DC',     ('D', 'C')),
    ('B9_DS',     ('D', 'S')),
    ('B10_CS',    ('C', 'S')),
    ('C1_QMD',    ('Q', 'M', 'D')),
    ('C2_QMC',    ('Q', 'M', 'C')),
    ('C3_QMS',    ('Q', 'M', 'S')),
    ('C4_QDC',    ('Q', 'D', 'C')),
    ('C5_QDS',    ('Q', 'D', 'S')),
    ('C6_QCS',    ('Q', 'C', 'S')),
    ('C7_MDC',    ('M', 'D', 'C')),
    ('C8_MDS',    ('M', 'D', 'S')),
    ('C9_MCS',    ('M', 'C', 'S')),
    ('C10_DCS',   ('D', 'C', 'S')),
    ('D1_QMDC',   ('Q', 'M', 'D', 'C')),
    ('D2_QMDS',   ('Q', 'M', 'D', 'S')),
    ('D3_QMCS',   ('Q', 'M', 'C', 'S')),
    ('D4_QDCS',   ('Q', 'D', 'C', 'S')),
    ('D5_MDCS',   ('M', 'D', 'C', 'S')),
    ('E1_QMDCS',  ('Q', 'M', 'D', 'C', 'S')),
]


MODULE_LABELS: Dict[str, str] = {
    'Q': 'QCNN',
    'M': '多域有标签监督',
    'D': 'DANN 域对抗',
    'C': 'CORAL',
    'S': 'Sparse',
}


def _make_ablation_preset(name: str, modules: Tuple[str, ...]) -> Dict[str, Any]:
    mods = set(modules)
    aux_losses = []
    if 'C' in mods:
        aux_losses.append(('coral', 'lambda_coral'))
    if 'S' in mods:
        aux_losses.append(('sparse', 'lambda_sparse'))

    combo = ' + '.join(MODULE_LABELS[m] for m in modules) if modules else '源域监督 CNN1D 基线'
    return {
        'modules':            list(modules),
        'backbone':           'qcnn' if 'Q' in mods else 'cnn1d',
        'use_target_domains': bool(mods & {'M', 'D', 'C', 'S'}),
        'use_multidomain':    'M' in mods,
        'use_dann':           'D' in mods,
        'aux_losses':         aux_losses,
        'description':        f"{name}: {combo}",
    }


ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    name: _make_ablation_preset(name, modules)
    for name, modules in ABLATION_COMBOS
}


# ──────────────────────────────────────────────────────────
# Comparison baselines: DAN / JAN / DANN / CDAN
# ──────────────────────────────────────────────────────────
# 仅追加对比实验配置，不改变任何已有消融 preset。
# train.py 默认在训练结束后自动调用 deploy.py 评测；只有显式传
# --no_auto_deploy 时才会跳过。

COMPARISON_PRESETS: Dict[str, Dict[str, Any]] = {
    'DAN': {
        'modules':            ['DAN'],
        'backbone':           'cnn1d',
        'use_target_domains': True,
        'use_multidomain':    False,
        'use_dann':           False,
        'use_cdan':           False,
        'aux_losses':         [('mmd', 'lambda_mmd')],
        'joint_aux_losses':   [],
        'description':        'DAN: source supervised CNN1D + MMD target alignment',
    },
    'JAN': {
        'modules':            ['JAN'],
        'backbone':           'cnn1d',
        'use_target_domains': True,
        'use_multidomain':    False,
        'use_dann':           False,
        'use_cdan':           False,
        'aux_losses':         [],
        'joint_aux_losses':   [('jan', 'lambda_mmd')],
        'description':        'JAN: source supervised CNN1D + joint MMD target alignment',
    },
    'DANN': {
        'modules':            ['DANN'],
        'backbone':           'cnn1d',
        'use_target_domains': True,
        'use_multidomain':    False,
        'use_dann':           True,
        'use_cdan':           False,
        'aux_losses':         [],
        'joint_aux_losses':   [],
        'description':        'DANN: source supervised CNN1D + GRL domain adversarial alignment',
    },
    'CDAN': {
        'modules':            ['CDAN'],
        'backbone':           'cnn1d',
        'use_target_domains': True,
        'use_multidomain':    False,
        'use_dann':           False,
        'use_cdan':           True,
        'aux_losses':         [],
        'joint_aux_losses':   [],
        'description':        'CDAN: source supervised CNN1D + conditional domain adversarial alignment',
    },
}

ABLATION_PRESETS.update(COMPARISON_PRESETS)


def get_preset(mode: str) -> Dict[str, Any]:
    if mode not in ABLATION_PRESETS:
        raise KeyError(f"未知 mode='{mode}'。可选：{list(ABLATION_PRESETS)}")
    return ABLATION_PRESETS[mode]


def list_modes() -> List[str]:
    return list(ABLATION_PRESETS)
