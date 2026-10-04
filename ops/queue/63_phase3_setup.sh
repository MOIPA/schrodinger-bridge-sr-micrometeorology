#!/bin/bash
# 63_phase3_setup.sh — 阶段 3 T3.5 前置门:AGL 探针(88)-> 生成配置(87)-> 通道校验(81)
#   -> 自检(95 损失 / 89 AGL 评估链路 / 80 数据集)-> 冒烟(p3_agl,AGL 表 + agl 损失,CPU 若干 iter)
# 前置依赖(并行任务产出,若缺则本脚本对应步会明确报 FAIL):
#   scripts/outline/88_phase3_probe.py        产出 results/phase3/probe.json
#   scripts/outline/87_gen_phase3_configs.py  读 --probe_json 生成
#       configs/深圳/phase3/config_wind_canvas_p3_{agl,joint,agllw}.yml
#   scripts/outline/99_phase3_tables.py       阶段 3 汇总表(63e 调)
# 既有链路自检:95_phase2_loss_fixture.py(C 任务可能已扩展覆盖 AGL 项)、
#   89_agl_eval_fixture.py(需真实 statics)、80_fixture_check.py。
# 冒烟说明:AGL 臂(agl_weight>0)不能用 scripts/smoke_wind_canvas.py(它不传 agl 表,
#   si.forward 会直接报"需要 agl=..."),故用内联冒烟:取 1 个 train batch,3 个 iter
#   前向 + 反向(si 传 AGL 插值表),CPU 上跑,打印各项损失分项。
# 手机执行: bash ops/queue/63_phase3_setup.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
git --no-pager log --oneline -1
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
EXP=ExperimentSchrodingerBridgeWindCanvas
OUT=ops/result/63_phase3_setup.txt
mkdir -p ops/result logs "configs/深圳/phase3" results/phase3
: > "$OUT"

{
echo "===== 0. 数据与 V* 基线检查 ====="
for d in prepare_npz_outline_fine prepare_npz_outline_coarse prepare_npz_outline_static; do
  if [ -d "$ROOT/$d" ]; then
    echo "[OK] $d ($(ls "$ROOT/$d" 2>/dev/null | wc -l) 个条目)"
  else
    echo "[缺] $d 不存在"
  fi
done
VSTAR="$ROOT/data/DL_result/$EXP/config_wind_canvas_p1r_t14_noenc_cos/checkpoint.pth"
if [ -f "$VSTAR" ]; then
  echo "[OK] V* 基线 checkpoint(阶段 1 最优 run r_t14_noenc_cos)存在"
else
  echo "[缺] V* 基线 checkpoint: $VSTAR(88 探针/89 自检可能需要)"
fi
for s in 88_phase3_probe 87_gen_phase3_configs 99_phase3_tables 95_phase2_loss_fixture \
         89_agl_eval_fixture 80_fixture_check; do
  if [ -f "scripts/outline/$s.py" ]; then
    echo "[OK] scripts/outline/$s.py"
  else
    echo "[缺] scripts/outline/$s.py(并行任务产物,先等它合入再跑本脚本相应步骤)"
  fi
done

echo ""
echo "===== 1. AGL 探针(88)-> results/phase3/probe.json ====="
$PY3D -u scripts/outline/88_phase3_probe.py
echo "88 exit=$?"
if [ -f results/phase3/probe.json ]; then
  echo "--- probe.json ---"
  cat results/phase3/probe.json
else
  echo "[FAIL] 未产生 results/phase3/probe.json;87 的 --probe_json 将无法继续"
fi

echo ""
echo "===== 2. 生成阶段 3 配置(87;读 probe.json) ====="
$PY3D -u scripts/outline/87_gen_phase3_configs.py --probe_json results/phase3/probe.json
echo "87 exit=$?"
ls -1 "configs/深圳/phase3" 2>/dev/null
for t in p3_agl p3_joint p3_agllw; do
  if [ -f "configs/深圳/phase3/config_wind_canvas_${t}.yml" ]; then
    echo "[OK] config_wind_canvas_${t}.yml"
  else
    echo "[警告] 缺 config_wind_canvas_${t}.yml(63b 对应 wave 会跳过该 tag)"
  fi
done

echo ""
echo "===== 3. 通道校验(81;阶段 3 目录) ====="
$PY3D -u scripts/outline/81_config_channel_check.py --config_dir "configs/深圳/phase3"
echo "81 exit=$?"

echo ""
echo "===== 4. 物理损失项合成自检(95;C 任务可能已扩展覆盖 AGL 项) ====="
$PY3D -u scripts/outline/95_phase2_loss_fixture.py
echo "95 exit=$?"

echo ""
echo "===== 5. AGL 评估链路合成自检(89;需真实 statics) ====="
$PY3D -u scripts/outline/89_agl_eval_fixture.py --static_dir "$ROOT/prepare_npz_outline_static"
echo "89 exit=$?"

echo ""
echo "===== 6. canvas 数据集合成自检(80;含 rho/AGL 表透传) ====="
$PY3D -u scripts/outline/80_fixture_check.py
echo "80 exit=$?"

echo ""
echo "===== 7. 冒烟(p3_agl:AGL 表 + agl 损失,3 iter,CPU) ====="
if [ -f "configs/深圳/phase3/config_wind_canvas_p3_agl.yml" ]; then
$PY3D -u - <<'PY'
import os
import sys

import torch

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
from src.utils.random_seed_helper import set_seeds

CFG = "configs/深圳/phase3/config_wind_canvas_p3_agl.yml"
EXP = "ExperimentSchrodingerBridgeWindCanvas"
cfg = load_config(EXP, CFG)
set_seeds(cfg.train.seed)
device = torch.device("cpu")
print("冒烟配置: agl_weight={} agl_replace_data={} return_agl_tables={}".format(
    cfg.si.agl_weight, cfg.si.agl_replace_data,
    getattr(cfg.data, "return_agl_tables", False)))
dl, _ = make_dataloaders_and_samplers(
    root_dir=ROOT, loader_config=cfg.loader, dataset_config=cfg.data,
    world_size=None, rank=None, train_valid_test_kinds=["train"])
it = iter(dl["train"])
net = make_model(cfg.model).to(device)
si = StochasticInterpolantFollmer(config=cfg.si, neural_net=net)
opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
for step in range(3):
    b = next(it)
    x, y, y0 = b["x"].to(device), b["y"].to(device), b["y0"].to(device)
    agl = None
    if "agl_idx_m" in b:
        agl = {"idx_m": b["agl_idx_m"].to(device), "w_m": b["agl_w_m"].to(device),
               "idx_i": b["agl_idx_i"].to(device), "w_i": b["agl_w_i"].to(device)}
    rho = b.get("rho")
    if rho is not None:
        rho = rho.to(device)
    out = si(y0=y0, y1=y, y_cond=x, rho=rho, agl=agl, return_parts=True)
    opt.zero_grad()
    out["total"].backward()
    opt.step()
    parts = {k: (None if v is None else round(float(v), 6)) for k, v in out.items()}
    print("iter {} total={:.6f} parts={}".format(step, float(out["total"]), parts))
    assert torch.isfinite(out["total"]).all(), "损失非有限值"
print("SMOKE OK")
PY
echo "smoke exit=$?"
else
  echo "[跳过] 冒烟:缺 configs/深圳/phase3/config_wind_canvas_p3_agl.yml(先看第 1/2 步)"
fi

echo ""
echo "===== 8. 后续步骤 ====="
cat <<'EOM'
(1) W1 训练(p3_agl + p3_joint):
      bash ops/queue/63b_phase3_waves.sh
(2) 查状态 / PEND 换队列:
      bash ops/queue/63c_check_phase3.sh
(3) W1 两臂训练完 -> 评估(90;含重评 p2_l1r2_lr2e4 回归):
      bash ops/queue/63d_eval_phase3.sh
(4) 闸门通过(前波评估产物存在)后 W2(p3_agllw):
      WAVE=2 bash ops/queue/63b_phase3_waves.sh
(5) 终评报告(99 表 + 98 空间):
      bash ops/queue/63e_phase3_report.sh
EOM
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result results/phase3
if [ -n "$(ls -A "configs/深圳/phase3" 2>/dev/null)" ]; then
  git add "configs/深圳/phase3"
fi
git commit -m "result 63 phase3 setup: probe + configs + fixture/smoke gates"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
