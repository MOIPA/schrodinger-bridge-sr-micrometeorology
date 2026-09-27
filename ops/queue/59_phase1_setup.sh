#!/bin/bash
# 59_phase1_setup.sh — 阶段 1 前置门:生成 15 个配置 -> 通道校验 -> 合成自检 -> 冒烟(base + t16)
# 手机执行: bash ops/queue/59_phase1_setup.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
OUT=ops/result/59_phase1_setup.txt
mkdir -p ops/result logs "configs/深圳/phase1" results/phase1
{
echo "===== 1. 生成 15 个阶段 1 配置(85) ====="
$PY3D -u scripts/outline/85_gen_phase1_configs.py
echo "===== 2. 通道校验(81) ====="
$PY3D -u scripts/outline/81_config_channel_check.py
echo "===== 3. canvas 数据集合成自检(80,含全部新输入组) ====="
$PY3D -u scripts/outline/80_fixture_check.py
echo "===== 4. AGL 评估链路合成自检(89) ====="
$PY3D -u scripts/outline/89_agl_eval_fixture.py --static_dir "$ROOT/prepare_npz_outline_static"
echo "===== 5. 冒烟 base(L1,无物理约束) ====="
$PY3D -u scripts/smoke_wind_canvas.py \
  --config_path "configs/深圳/phase1/config_wind_canvas_p1_base.yml" --device cpu --backward
echo "smoke base exit=$?"
echo "===== 6. 冒烟 t16_residual(L1 + 残差输出) ====="
$PY3D -u scripts/smoke_wind_canvas.py \
  --config_path "configs/深圳/phase1/config_wind_canvas_p1_t16_residual.yml" --device cpu --backward
echo "smoke t16 exit=$?"
} > "$OUT" 2>&1
tail -40 "$OUT"
git add ops/result "configs/深圳/phase1"
git commit -m "result 59 phase1 setup: 15 configs + fixture/smoke gates"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
