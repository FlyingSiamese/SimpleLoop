# SimpleLoop

Looped Transformer 的个人复现实验。

读完 Yang et al. 2024 (Looped Transformers are Better at Learning Learning Algorithms) 之后想自己跑一遍。做法是在合成线性回归任务上训一个 2 层 Transformer，参数在 8 次 loop 之间共享，看循环迭代对预测误差有什么影响，再跟算力对齐的普通 Transformer 和最小二乘比一比。

## 环境

```bash
conda create -n simpleloop python=3.11 -y
conda activate simpleloop
pip install -r requirements.txt
```

## 跑起来

先生成数据（10 万 / 5 千 / 5 千）：

```bash
python -m src.generate_data --config src/configs/base.yaml
```

训练。base.yaml 里 `loop_layers: 2` 是每轮几层，`baseline_layers: 16` 是 baseline 的层数（2×8=16，跟 looped 算力对齐）：

```bash
python -m src.train          --config src/configs/base.yaml --run-name main
python -m src.train_baseline --config src/configs/base.yaml --run-name baseline
```

消融 5 组，run 名就用配置文件名去掉后缀：

```bash
python -m src.train --config src/configs/ablation_no_injection.yaml --run-name ablation_no_injection
python -m src.train --config src/configs/ablation_loop_4.yaml       --run-name ablation_loop_4
python -m src.train --config src/configs/ablation_loop_12.yaml      --run-name ablation_loop_12
python -m src.train --config src/configs/ablation_window_1.yaml     --run-name ablation_window_1
python -m src.train --config src/configs/ablation_window_8.yaml     --run-name ablation_window_8
```

嫌麻烦可以一把梭（7 个 run 串行，约 6.5 小时，中途断了重跑会跳过已完成的）：

```bash
nohup ./run_all.sh > outputs/run_all.log 2>&1 &
```

评测和出图：

```bash
python -m src.evaluate     --config src/configs/base.yaml --checkpoint outputs/main/checkpoints/best.pt
python -m src.evaluate_ood --config src/configs/base.yaml --checkpoint outputs/main/checkpoints/best.pt
python -m src.evaluate_ablation
python -m src.plot_results --run outputs/main
```

## 结果

跑完的结果和图表放在 `results/`，训练产物在 `outputs/`（15G，没进仓库）。

**最值得看的一张：** `results/figures/loop_count_ablation.png`

不同训练 loop 数下，测试误差随推理 loop 次数的变化：

| 训练时用几轮 | 误差最低点 | 最低 MSE | 第 32 轮 MSE |
|---|---|---|---|
| 4 | 第 4 轮 | 0.00802 | 0.10180 |
| 8 | 第 7 轮 | 0.00704 | 0.06301 |
| 12 | 第 10 轮 | 0.00602 | 0.02927 |

拐点差不多就落在训练时用的 loop 数上，之后开始缓慢变差。训练时多绕几轮，平台更低，发散也更慢。这个跟论文 Fig. 4 说的对得上。

**跟 baseline 比：**

| | 参数量 | test MSE | R² |
|---|---|---|---|
| Looped (2 层 × 8 loops) | 2,103,553 | 0.06301 | 0.938 |
| 普通 Transformer (16 层) | 16,790,785 | 0.02020 | 0.980 |
| 最小二乘 | — | 1.0e-13 | 1.000 |

算力一样，但 16 层的 baseline 好 3 倍。用 1/8 的参数没追上。论文里其实也承认了这点（"Nevertheless, it fails to replicate the performance of the standard transformer"）。

**其它几组：**

- loss window：T=8（监督全部 8 轮）最好，收敛最快、发散最慢；T=1 只监督最后一轮，前 5 轮几乎不动，第 6~8 轮才突然掉下去，有点像"憋到最后一次性算完"
- input injection：开着能让前几轮收敛更快（第 3 轮 0.033 vs 0.083），但长期发散速度差不多（8.9× vs 6.6×），说不上是稳定性的关键
- OOD：输入尺度最脆弱（s=3 时 MSE 从 0.06 涨到 4.7），斜协方差还行（0.062 → 0.078），标签噪声单调退化，context 只在训练的 k=32 上有效，k≤4 时比直接输出 0 还差

## 踩过的坑

**1. 损失函数少监督了 32 倍**

第一版只监督最后一个 query，结果 train loss 掉到 0.002、val loss 反而涨到 1.8，测试 R² = −0.002。模型把 10 万个训练 task 全背下来了，完全没泛化。

后来翻论文 Eq.1 才发现，它对**所有 prompt 前缀**求平均：

$$\frac{1}{k+1}\sum_{i=0}^{k}\ell\left(Y_t(P^i),\ f(x_{i+1})\right)$$

也就是一个 task 有 k+1=33 个预测，不是 1 个。改完之后 R² 直接到 0.938。

实现上不用改模型结构——prompt 是 `Ex(x1) Ey(y1) ... Ex(xk) Ey(yk) Ex(xq)` 交替排的，所有偶数位置正好就是这 33 个前缀的输出位置，`H[:, ::2]` 取出来跟 `y` 一一对应。

**2. YAML 把 2e-4 解析成字符串**

`lr: 2e-4` 不是 float 是 `str`，训练到 `AdamW` 那里才炸。必须写成 `0.0002` 或 `2.0e-4`。类似地 `key:value` 不加空格也不是键值对。

**3. `.squeeze(-1)` 和广播**

验证脚本里 `(xq.unsqueeze(1) @ w_hat).squeeze(-1)` 出来是 `[N,1]` 不是 `[N]`，减 `[N]` 的 target 会广播成 `[N,N]`，MSE 算出来是个看着挺正常的数（≈2，正好是 y 的方差的两倍）。现在训练循环里加了形状断言。

## 文件

```
src/configs/          配置（base + 5 组消融）
src/                  模型、数据、训练、评测
src/train*.py         训练入口
src/evaluate*.py      评测入口
src/plot_results.py   出图
run_all.sh            串行跑完全部实验
results/              跑完的结果与图表
data/                 数据集（未进仓库）
outputs/              训练产物（未进仓库）
```

## 参考

Yang et al. 2024. Looped Transformers are Better at Learning Learning Algorithms. ICLR 2024.
