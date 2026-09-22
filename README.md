# SimpleLoop

自建 Looped Transformer，在合成线性回归任务上做 in-context learning，评估 loop 迭代对预测误差的影响。

## 结果

7 个 run，每个 20000 步，RTX 4060 Ti 单卡约 6.5 小时。完整数据与图表见 `results/`。

### 核心发现：发散点精确跟着训练 loop 数移动

| 配置 | 训练 b | 误差最低点 | 最低 MSE | 第 32 轮 MSE |
|---|---|---|---|---|
| `ablation_loop_4` | 4 | 第 **4** 轮 | 0.00802 | 0.10180 |
| `main` | 8 | 第 **7** 轮 | 0.00704 | 0.06301 |
| `ablation_loop_12` | 12 | 第 **10** 轮 | 0.00602 | 0.02927 |

模型在训练 loop 数之内持续改进（误差降两个数量级），**超过之后单调缓慢发散**。b 越大，平台越低、发散越慢。这复现了论文 Fig. 4 的现象。

### 消融

| 消融 | 结果 |
|---|---|
| **loss window** T=1 / 4 / 8 | T=8 全面最优（收敛最快、发散最慢）；T=1 退化成"憋到最后一轮一次性算完" |
| **input injection** 开 / 关 | 开启加速早期收敛（第 3 轮 0.033 vs 0.083），但**长期发散速率接近**（8.9× vs 6.6×），非长期稳定性关键 |
| **baseline** 16 层（算力对齐） | **16 层更好**：MSE 0.0202 vs looped 0.0630，R² 0.980 vs 0.938。looped 用 1/8 参数未能追平 |

### 主要指标

```
Looped  2 layers x 8 loops   参数量  2,103,553    test MSE 0.06301   R² 0.938
Baseline 16 layers           参数量 16,790,785    test MSE 0.02020   R² 0.980
Least Squares                                       test MSE 1.0e-13   R² 1.000
```

### OOD

```
输入尺度    s=1.0 → 0.062    s=3.0 → 4.716     脆弱，无尺度不变性
斜协方差    identity 0.062 → skewed 0.078      鲁棒（仅恶化 26%）
标签噪声    sigma=0 → 0.062  sigma=0.5 → 0.528  单调退化
context     k=32 → 0.063    k≤4 → >1.0         只在训练的 k 上有效
```

### 关键图表

`results/figures/` 下 11 张，其中：

- `loop_count_ablation.png` —— 三条曲线的拐点落在 4 / 7 / 10，最有价值
- `mse_vs_loop.png` —— 主实验的 1~32 轮完整曲线
- `loss_window_ablation.png` —— T=1 的"末轮发力"形态对比 T=8 的"逐轮改进"

### 产物结构

```
results/
├── figures/            11 张图
├── main/               evaluation.json + ood.json + metrics.jsonl
├── baseline/           同上
├── ablations/          5 组消融的 metrics.jsonl
└── ablation.json       消融汇总
```

## 环境

```bash
conda create -n simpleloop python=3.11 -y
conda activate simpleloop
pip install -r requirements.txt
```

## 一键跑完全部实验

```bash
conda activate simpleloop
nohup ./run_all.sh > outputs/run_all.log 2>&1 &
```

串行执行：数据 → 训练 7 个 run → 评测 → 消融汇总 → 出图，约 6.5 小时。

- 已存在 `outputs/<run>/checkpoints/best.pt` 的训练自动跳过，**中断后直接重跑即可续上**
- `./run_all.sh --force` 忽略已有 checkpoint 全部重跑
- 收尾会列出失败的步骤；只要有一个失败，退出码为 1

## 数据

```bash
python -m src.generate_data --config src/configs/base.yaml
```

生成 `data/train.pt`(100000)、`val.pt`(5000)、`test.pt`(5000)、`metadata.json`。

## 训练

全部命令在项目根目录执行。

```bash
python -m src.train          --config src/configs/base.yaml --run-name main
python -m src.train_baseline --config src/configs/base.yaml --run-name baseline
```

消融实验 5 组，run 名取配置文件名去掉 `.yaml`：

```bash
python -m src.train --config src/configs/ablation_no_injection.yaml --run-name ablation_no_injection
python -m src.train --config src/configs/ablation_loop_4.yaml       --run-name ablation_loop_4
python -m src.train --config src/configs/ablation_loop_12.yaml      --run-name ablation_loop_12
python -m src.train --config src/configs/ablation_window_1.yaml     --run-name ablation_window_1
python -m src.train --config src/configs/ablation_window_8.yaml     --run-name ablation_window_8
```

续训：

```bash
python -m src.train --config src/configs/base.yaml --run-name main \
  --resume outputs/main/checkpoints/step_010000.pt
```

## 评测

```bash
python -m src.evaluate \
  --config src/configs/base.yaml \
  --checkpoint outputs/main/checkpoints/best.pt

python -m src.evaluate_ood \
  --config src/configs/base.yaml \
  --checkpoint outputs/main/checkpoints/best.pt

python -m src.evaluate_ablation
```

`--arch` 默认 `auto`，从 checkpoint 的层数自动判断 looped / baseline，无需手填。

## 出图

```bash
python -m src.plot_results --run outputs/main
```

生成 11 张图到 `outputs/main/figures/`。缺少某个 json 时自动跳过对应的图。

## 配置

| 文件 | 相对 base.yaml 的差异 |
|---|---|
| `base.yaml` | 主实验，兼作所有消融的对照点 |
| `ablation_no_injection.yaml` | `input_injection: false` |
| `ablation_loop_4.yaml` / `ablation_loop_12.yaml` | `train_loops: 4` / `12` |
| `ablation_window_1.yaml` / `ablation_window_8.yaml` | `loss_window: 1` / `8` |

`model.loop_layers` 与 `model.baseline_layers` 写在同一份配置里：looped 用前者，baseline 用后者（16 层，与 2 layers × 8 loops 算力对齐）。同一份配置保证两者的 lr、优化器、batch size、训练步数不可能不一致。

YAML 的科学计数法必须带小数点：`0.0002` 或 `2.0e-4` 是 float，`2e-4` 会被解析成字符串。

## 目录

```
src/configs/         配置与加载
src/                 模型、数据、训练、评测模块
src/train*.py        训练入口
src/evaluate*.py     评测入口
src/plot_results.py  出图入口
data/                数据集
outputs/             训练与评测产物
```
