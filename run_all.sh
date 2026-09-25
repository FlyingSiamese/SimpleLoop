#!/usr/bin/env bash
# 串行跑完全部实验：数据 -> 训练 7 个 run -> 评测 -> 消融汇总 -> 出图
#
#   conda activate simpleloop
#   nohup ./run_all.sh > outputs/run_all.log 2>&1 &      # 后台跑
#   ./run_all.sh                                          # 前台跑
#   ./run_all.sh --force                                  # 忽略已有 checkpoint，全部重跑
#
# 已存在 outputs/<run>/checkpoints/best.pt 的训练会自动跳过，
# 所以中断后直接重跑本脚本即可续上，不会重复劳动。
#
# 预计总时长约 6.5 小时（RTX 4060 Ti，batch 256）：
#   main + baseline                约 1.8 h
#   5 组消融                       约 4.5 h
#   评测 + 出图                    约 5 min

set -u

cd "$(dirname "$0")"

if ! python -c "import torch" 2>/dev/null; then
    echo "错误：当前 python 环境没有 torch。请先 conda activate simpleloop" >&2
    exit 1
fi

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

FAILED=()

step() { echo; echo "===== [$(date '+%F %T')] $* ====="; }

should_run() {
    [ "$FORCE" -eq 1 ] && return 0
    [ ! -f "outputs/$1/checkpoints/best.pt" ]
}

# ---------------------------------------------------------------- 1. 数据
if [ ! -f data/train.pt ]; then
    step "生成数据"
    python -u -m src.generate_data --config src/configs/base.yaml || FAILED+=("generate_data")
else
    step "跳过数据生成（data/train.pt 已存在）"
fi

# ------------------------------------------------- 2. 训练（looped 主实验 + 消融）
RUNS="src/configs/base.yaml main
src/configs/ablation_no_injection.yaml ablation_no_injection
src/configs/ablation_loop_4.yaml ablation_loop_4
src/configs/ablation_loop_12.yaml ablation_loop_12
src/configs/ablation_window_1.yaml ablation_window_1
src/configs/ablation_window_8.yaml ablation_window_8"

while read -r cfg run; do
    [ -z "${cfg:-}" ] && continue
    if should_run "$run"; then
        step "训练 $run   <- $cfg"
        python -u -m src.train --config "$cfg" --run-name "$run" || FAILED+=("train:$run")
    else
        step "跳过训练 $run（已存在 best.pt）"
    fi
done <<< "$RUNS"

# --------------------------------------------------- 3. 训练算力对齐 baseline
if should_run baseline; then
    step "训练 baseline   <- src/configs/base.yaml"
    python -u -m src.train_baseline --config src/configs/base.yaml --run-name baseline \
        || FAILED+=("train:baseline")
else
    step "跳过训练 baseline（已存在 best.pt）"
fi

# ------------------------------------------------------------------ 4. 评测
run_eval() {
    local name="$1" ckpt="$2"
    if [ ! -f "$ckpt" ]; then
        step "跳过 $name 评测（缺 $ckpt）"
        return
    fi
    step "ID 评测 $name"
    python -u -m src.evaluate --config src/configs/base.yaml --checkpoint "$ckpt" \
        || FAILED+=("evaluate:$name")
    if [ "$name" = main ]; then
        step "ID 评测 main（训练轮数：8）"
        python -u -m src.evaluate --config src/configs/base.yaml \
            --checkpoint "$ckpt" --num-loops 8 \
            || FAILED+=("evaluate:main:8")
    fi
    step "OOD 评测 $name"
    python -u -m src.evaluate_ood --config src/configs/base.yaml --checkpoint "$ckpt" \
        || FAILED+=("evaluate_ood:$name")
    if [ "$name" = main ]; then
        step "OOD 评测 main（训练轮数：8）"
        python -u -m src.evaluate_ood --config src/configs/base.yaml \
            --checkpoint "$ckpt" --num-loops 8 \
            || FAILED+=("evaluate_ood:main:8")
    fi
}

run_eval main     outputs/main/checkpoints/best.pt
run_eval baseline outputs/baseline/checkpoints/best.pt

# ------------------------------------------------------------ 5. 消融与出图
step "消融汇总"
python -u -m src.evaluate_ablation || FAILED+=("evaluate_ablation")

step "出图"
python -u -m src.plot_results --run outputs/main || FAILED+=("plot_results")

# ------------------------------------------------------------------ 6. 汇报
echo
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "===== [$(date '+%F %T')] 全部完成 ====="
    echo "图表: outputs/main/figures/"
else
    echo "===== [$(date '+%F %T')] 以下步骤失败 ====="
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    echo
    echo "修好后重跑本脚本，已完成的训练会自动跳过。"
    exit 1
fi
