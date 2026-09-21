# SimpleLoop

自建 Looped Transformer，在合成线性回归任务上做 in-context learning，评估 loop 迭代对预测误差的影响。

## 环境

```bash
conda create -n simpleloop python=3.11 -y
conda activate simpleloop
pip install -r requirements.txt
```

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
