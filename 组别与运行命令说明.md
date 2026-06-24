# 组别与运行命令说明

本文档记录表 2 中消融组别与代码实际 `mode` 的对应关系，并整理训练、部署、混淆矩阵、聚类图和方法对比图的常用终端命令。
figure和figure_out包含论文中所有图像绘制脚本 训练曲线和矩阵在result对应目录下、cache是数据库预处理数据 方便多次处理相同数据，可忽略，当使用对应加噪情况时，代码会自动读取已预处理的文件，若没有预处理第一次加载会比较久
所有命令默认在项目总目录执行：

```powershell
cd E:\Python\Multi-DANN
```

## 1. 表 3.2 组别与代码 mode 对应关系

代码中三个核心开关含义如下：

| 代码字母 | 对应模块 | 代码开关 |
|---|---|---|
| `M` | Multi-domain，多域有标签监督 | `use_multidomain=True` |
| `D` | DANN，域对抗训练 | `use_dann=True` |
| `C` | CORAL，协方差对齐损失 | `aux_losses=[("coral", "lambda_coral")]` |

注意：论文表格中的 `A1/A2/.../C1` 是表 3.2 的组别编号；代码里的 `A1_Q/A2_M/C7_MDC` 是内部消融枚举编号，二者不一定同名。实际运行时按下面的 `代码实际 mode` 使用。其余组别是废弃实验组别。

| 表 3.2 group | DANN | CORAL | Multi-domain | 消融模块数 | 代码实际 mode | 实际模块组合 |
|---|---:|---:|---:|---:|---|---|
| A1 | √ | √ | √ | 0 | `C7_MDC` | `M + D + C` |
| A2 | × | √ | √ | 1 | `B6_MC` | `M + C` |
| A3 | √ | × | √ | 1 | `B5_MD` | `M + D` |
| A4 | √ | √ | × | 1 | `B8_DC` | `D + C` |
| B1 | √ | × | × | 2 | `A3_D` | `D` |
| B2 | × | × | √ | 2 | `A2_M` | `M` |
| B3 | × | √ | × | 2 | `A4_C` | `C` |
| C1 | × | × | × | 3 | `A0_BASE` | CNN1D baseline |

如果只跑表 3.2 这 8 组，推荐按这个顺序：

```text
C7_MDC,B6_MC,B5_MD,B8_DC,A3_D,A2_M,A4_C,A0_BASE
```

当前仓库分支为 `exp/ablation`，所以默认结果目录格式为：

```text
results\exp_ablation_<DATASET>_<MODE>_seed<SEED>
```

例如：

```text
results\exp_ablation_PU_C7_MDC_seed1000
```

## 2. 训练命令

查看所有可用 `mode`：

```powershell
python train.py --list
```

训练单个组别，并在训练结束后自动部署、写入 Excel：

```powershell
python train.py --mode C7_MDC --dataset PU --seed 1000 --excel results\deploy_metrics.xlsx
```

只训练，不自动部署：

```powershell
python train.py --mode C7_MDC --dataset PU --seed 1000 --no_auto_deploy
```

主要使用：&&&一次训练表 3.2 的全部 8 个组别，5 个随机种子，***--excel后接想要将性能指标保存的excel文件的位置***&&&：

```powershell
python train.py --modes C7_MDC,B6_MC,B5_MD,B8_DC,A3_D,A2_M,A4_C,A0_BASE --dataset PU --seeds 42,100,200,400,1000 --excel results\deploy_metrics.xlsx
```

CWRU 数据集对应命令：

```powershell
python train.py --modes C7_MDC,B6_MC,B5_MD,B8_DC,A3_D,A2_M,A4_C,A0_BASE --dataset CWRU --seeds 42,100,200,400,1000 --excel results\deploy_metrics_CWRU.xlsx
```

单域训练示例，只用 `-6dB` 域训练：

```powershell
python train.py --mode C7_MDC --dataset PU --seed 1000 --single_domain=-6 --no_auto_deploy
```

训练后主要保存位置：

| 文件 | 保存位置 | 含义 |
|---|---|---|
| `checkpoint.pth` | `results\exp_ablation_PU_C7_MDC_seed1000\checkpoint.pth` | 最优模型权重 |
| `train_curves.png` | `results\exp_ablation_PU_C7_MDC_seed1000\train_curves.png` | 训练/验证 loss 与 accuracy 曲线 |
| `train_history.json` | 同一结果目录 | 训练曲线原始数据 |
| `train_log.txt` | 同一结果目录 | 训练日志 |
| `config.json` | 同一结果目录 | 本次实验配置 |
| `preset.json` | 同一结果目录 | 当前 mode 的模块开关 |

## 3. 部署与评测命令

单独部署已有模型，在 test 集上评测并绘制混淆矩阵：

```powershell
python deploy.py --mode C7_MDC --dataset PU --seed 1000 --excel results\deploy_metrics.xlsx
```

指定已有结果目录和权重文件进行部署：

```powershell
python deploy.py --run_dir results\exp_ablation_PU_C7_MDC_seed1000 --ckpt results\exp_ablation_PU_C7_MDC_seed1000\checkpoint.pth --excel results\deploy_metrics.xlsx
```

只从已有 `confusion_matrices.npz` 计算 Acc/F1/FDR/FPR，不重新推理：

```powershell
python deploy_metrics.py --from_npz --run_dir results\exp_ablation_PU_C7_MDC_seed1000 --excel results\deploy_metrics.xlsx
```

汇总某个 mode 的多个 seed 到 Excel：

```powershell
python deploy_metrics.py --aggregate_existing --dataset PU --mode C7_MDC --excel results\deploy_metrics.xlsx
```

部署后主要保存位置：

| 文件 | 保存位置 | 含义 |
|---|---|---|
| `deploy_results.txt` | `results\exp_ablation_PU_C7_MDC_seed1000\deploy_results.txt` | 每个 SNR 域的 Acc/F1/FDR/FPR |
| `deploy_results.json` | 同一结果目录 | 部署结果 JSON |
| `deploy_metrics.json` | 同一结果目录 | 指标 JSON |
| `confusion_matrices.npz` | 同一结果目录 | 混淆矩阵原始数据 |
| `confusion_matrices\*.png` | `results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\` | 混淆矩阵图 |

## 4. 各绘图脚本命令与保存位置

### 4.1 训练曲线图

绘图位置：`trainer.py` 中的训练曲线绘图函数。

终端命令：不单独运行 `trainer.py`，训练时由 `train.py` 自动生成。

```powershell
python train.py --mode C7_MDC --dataset PU --seed 1000
```

保存位置：

```text
results\exp_ablation_PU_C7_MDC_seed1000\train_curves.png
```

### 4.2 混淆矩阵图：deploy.py

正常部署并绘制混淆矩阵：

```powershell
python deploy.py --mode C7_MDC --dataset PU --seed 1000
```

不跑模型，只根据已有 `confusion_matrices.npz` 重绘混淆矩阵：

```powershell
python deploy.py --plot_only --run_dir results\exp_ablation_PU_C7_MDC_seed1000
```

保存位置：

```text
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\cm_p6dB.png
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\cm_p3dB.png
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\cm_p0dB.png
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\cm_n3dB.png
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\cm_n6dB.png
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\grid.png
```

### 4.3 混淆矩阵图：plot_diagnostics.py

`plot_diagnostics.py` 也可以单独重绘混淆矩阵：

```powershell
python figure\plot_diagnostics.py --kind confusion --mode C7_MDC --dataset PU --seed 1000 --run_dir results\exp_ablation_PU_C7_MDC_seed1000
```

保存位置同上：

```text
results\exp_ablation_PU_C7_MDC_seed1000\confusion_matrices\
```

### 4.4 单独绘制一个聚类图：plot_diagnostics.py

绘制 `-6dB` 测试域的 KMeans/t-SNE 聚类图：

```powershell
python figure\plot_diagnostics.py --kind kmeans --mode C7_MDC --dataset PU --seed 1000 --run_dir results\exp_ablation_PU_C7_MDC_seed1000 --snr -6 --pu_class_set 10class
```

绘制所有 SNR 域合并后的聚类图：

```powershell
python figure\plot_diagnostics.py --kind kmeans --mode C7_MDC --dataset PU --seed 1000 --run_dir results\exp_ablation_PU_C7_MDC_seed1000 --all_snrs --pu_class_set 10class
```

如果想用 PCA 而不是 t-SNE：

```powershell
python figure\plot_diagnostics.py --kind kmeans --mode C7_MDC --dataset PU --seed 1000 --run_dir results\exp_ablation_PU_C7_MDC_seed1000 --snr -6 --embed pca --pu_class_set 10class
```

保存位置：

```text
results\exp_ablation_PU_C7_MDC_seed1000\diagnostic_plots\PU_kmeans_n6dB.png
results\exp_ablation_PU_C7_MDC_seed1000\diagnostic_plots\PU_kmeans_n6dB.svg
results\exp_ablation_PU_C7_MDC_seed1000\diagnostic_plots\PU_kmeans_n6dB.pdf
results\exp_ablation_PU_C7_MDC_seed1000\diagnostic_plots\PU_kmeans_n6dB.json
```

`--all_snrs` 时文件名为：

```text
results\exp_ablation_PU_C7_MDC_seed1000\diagnostic_plots\PU_kmeans_all_snrs.*
```

### 4.5 同时绘制两个聚类图：plot_diagnostics.py

左右并排比较两个模型的聚类效果，例如 `C7_MDC` 与 `DANN`：

```powershell
python figure\plot_diagnostics.py --kind compare_kmeans --compare_modes C7_MDC DANN --dataset PU --seed 1000 --snr -6 --pu_class_set 10class
```

比较表 3.2 中完整模型和基线模型：

```powershell
python figure\plot_diagnostics.py --kind compare_kmeans --compare_modes C7_MDC A0_BASE --dataset PU --seed 1000 --snr -6 --pu_class_set 10class
```

保存位置：

```text
figure_output\diagnostic_compare_plots\PU_kmeans_compare_C7_MDC_vs_DANN_n6dB.png
figure_output\diagnostic_compare_plots\PU_kmeans_compare_C7_MDC_vs_DANN_n6dB.svg
figure_output\diagnostic_compare_plots\PU_kmeans_compare_C7_MDC_vs_DANN_n6dB.pdf
figure_output\diagnostic_compare_plots\PU_kmeans_compare_C7_MDC_vs_DANN_n6dB.json
```

### 4.6 方法对比折线图：plot_method_comparison.py

该脚本读取：

```text
results\deploy_metrics.xlsx
results\deploy_metrics_CWRU.xlsx
```

要求 Excel 的 `deploy_metrics` sheet 中已经有 `DAN/JAN/DANN/CDAN/C7_MDC` 的 `MEAN+/-STD` 汇总行。

绘图命令：

```powershell
python figure\plot_method_comparison.py
```

保存位置：

```text
figure_output\method_comparison_figures\PU_method_comparison_acc.png
figure_output\method_comparison_figures\PU_method_comparison_acc.svg
figure_output\method_comparison_figures\PU_method_comparison_acc.pdf
figure_output\method_comparison_figures\PU_method_comparison_acc.tiff
figure_output\method_comparison_figures\CWRU_method_comparison_acc.*
figure_output\method_comparison_figures\PU_CWRU_method_comparison_acc.*
```

### 4.7 噪声遮蔽图：fig_noise_masking.py

绘图命令：

```powershell
python figure\fig_noise_masking.py
```

保存位置：

```text
figure_output\noise_masking.png
figure_output\noise_masking.pdf
```

### 4.8 CWRU/PU 信号对比图：figure_signal_comparison.py

绘图命令：

```powershell
python figure\figure_signal_comparison.py
```

保存位置：

```text
figure_output\signal_comparison.png
figure_output\signal_comparison.pdf
```




# 5. 常用一键流程

完整跑一个组别：训练、自动部署、生成混淆矩阵、写 Excel。

```powershell
python train.py --mode C7_MDC --dataset PU --seed 1000 --excel results\deploy_metrics.xlsx
```

训练已经完成，只补部署和混淆矩阵：

```powershell
python deploy.py --mode C7_MDC --dataset PU --seed 1000 --excel results\deploy_metrics.xlsx
```

部署已经完成，只重画混淆矩阵：

```powershell
python deploy.py --plot_only --run_dir results\exp_ablation_PU_C7_MDC_seed1000
```

部署已经完成，只画聚类图：

```powershell
python figure\plot_diagnostics.py --kind kmeans --mode C7_MDC --dataset PU --seed 1000 --run_dir results\exp_ablation_PU_C7_MDC_seed1000 --snr -6 --pu_class_set 10class
```

部署已经完成，只画两个模型的聚类对比：

```powershell
python figure\plot_diagnostics.py --kind compare_kmeans --compare_modes C7_MDC DANN --dataset PU --seed 1000 --snr -6 --pu_class_set 10class
```
