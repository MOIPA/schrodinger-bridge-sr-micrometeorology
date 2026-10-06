#!/bin/bash
# 64_arch_setup.sh — 阶段 3 架构对比 T3.1-3.3 前置门:
#   生成配置(83)-> 通道校验(81,si/reg 配置;edm 配置由 83 剥离校验)
#   -> 自检(82 架构 A/B/C / 80 数据集 / 89 AGL 评估链路 / 95 阶段 2 损失)
#   -> 冒烟(swin 训练 / reg 回归训练 / edm 订正器假回归,各 3 iter 内 CPU,1 个真实 batch)
# 前置依赖(先合入再跑):
#   scripts/outline/{83_gen_arch_configs,82_arch_fixture,84_ensemble_eval}.py
#   src/dl_model/{swinir_arch,regression_wrapper,edm_correction}.py
#   src/dl_train/reg_optim_helper.py scripts/{train_regression_model,train_edm_correction}.py
# 数据(与阶段 1-3 同):prepare_npz_outline_{fine,coarse,static} + 参考臂 checkpoint
#   data/DL_result/ExperimentSchrodingerBridgeWindCanvas/config_wind_canvas_p2_l1r2_lr2e4/checkpoint.pth
#   (64d 参考臂 ens16 重评需要)。
# 手机执行: bash ops/queue/64_arch_setup.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
git --no-pager log --oneline -1
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
EXP=ExperimentSchrodingerBridgeWindCanvas
OUT=ops/result/64_arch_setup.txt
mkdir -p ops/result logs "configs/深圳/phase3_arch" results/phase3_arch
: > "$OUT"

{
echo "===== 0. 前置文件与数据检查 ====="
for f in scripts/outline/83_gen_arch_configs.py scripts/outline/82_arch_fixture.py \
         scripts/outline/84_ensemble_eval.py scripts/outline/81_config_channel_check.py \
         scripts/outline/80_fixture_check.py scripts/outline/89_agl_eval_fixture.py \
         scripts/outline/95_phase2_loss_fixture.py scripts/outline/99_phase3_tables.py \
         src/dl_model/swinir_arch.py src/dl_model/regression_wrapper.py \
         src/dl_model/edm_correction.py src/dl_train/reg_optim_helper.py \
         scripts/train_regression_model.py scripts/train_edm_correction.py; do
  if [ -f "$f" ]; then echo "[OK] $f"; else echo "[缺] $f(先合入对应任务产物)"; fi
done
for d in prepare_npz_outline_fine prepare_npz_outline_coarse prepare_npz_outline_static; do
  if [ -d "$ROOT/$d" ]; then
    echo "[OK] $d ($(ls "$ROOT/$d" 2>/dev/null | wc -l) 个条目)"
  else
    echo "[缺] $d 不存在"
  fi
done
REF_CK="$ROOT/data/DL_result/$EXP/config_wind_canvas_p2_l1r2_lr2e4/checkpoint.pth"
[ -f "$REF_CK" ] && echo "[OK] 参考臂 checkpoint(64d ens16 重评用)" \
  || echo "[缺] 参考臂 checkpoint: $REF_CK(64d 会跳过参考臂重评)"

echo ""
echo "===== 1. 生成架构配置(83;逐键 diff 见下) ====="
$PY3D -u scripts/outline/83_gen_arch_configs.py
echo "83 exit=$?"
ls -1 "configs/深圳/phase3_arch" 2>/dev/null
for t in p3_arch_unet_s2 p3_arch_swin p3_arch_swin_s2 p3_arch_reg p3_arch_reg_s2 \
         p3_arch_edm p3_arch_edm_s2; do
  [ -f "configs/深圳/phase3_arch/config_wind_canvas_${t}.yml" ] \
    && echo "[OK] config_wind_canvas_${t}.yml" || echo "[警告] 缺 config_wind_canvas_${t}.yml"
done

echo ""
echo "===== 2. 通道校验(81;仅 si/reg 配置) ====="
echo "说明:81 全目录模式会在 edm 配置上报 KeyError: 'edm'——base loader 按设计读不了"
echo "      edm/reg 顶段(训练脚本 load_edm_config 剥离后才行);83 已用 load_edm_config"
echo "      对 edm 配置做等价通道校验(ctx=2*out+in)与字段核对。"
$PY3D -u scripts/outline/81_config_channel_check.py --config_dir "configs/深圳/phase3_arch" \
  --glob "config_wind_canvas_p3_arch_[rsu]*.yml"
echo "81 exit=$?"

echo ""
echo "===== 3. 架构合成自检(82;A swin / B 回归 / C EDM / D 84 指标 / E 配置) ====="
$PY3D -u scripts/outline/82_arch_fixture.py
echo "82 exit=$?"

echo ""
echo "===== 4. canvas 数据集合成自检(80) ====="
$PY3D -u scripts/outline/80_fixture_check.py
echo "80 exit=$?"

echo ""
echo "===== 5. AGL 评估链路合成自检(89;需真实 statics) ====="
$PY3D -u scripts/outline/89_agl_eval_fixture.py --static_dir "$ROOT/prepare_npz_outline_static"
echo "89 exit=$?"

echo ""
echo "===== 6. 阶段 2 损失链合成自检(95) ====="
$PY3D -u scripts/outline/95_phase2_loss_fixture.py
echo "95 exit=$?"

echo ""
echo "===== 7. 冒烟(swin SI / reg 回归 / edm 假回归;3 iter,CPU,batch=1) ====="
$PY3D -u - <<'PY'
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F
import yaml

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.edm_correction import EDMCorrector, EDMCorrectorConfig
from src.dl_model.model_maker import make_model
from src.dl_model.regression_wrapper import RegressionModel
from src.dl_model.si_follmer.si_follmer_framework import StochasticInterpolantFollmer
from src.utils.random_seed_helper import set_seeds

EXP = "ExperimentSchrodingerBridgeWindCanvas"
CFG_DIR = "configs/深圳/phase3_arch"
DEV = torch.device("cpu")


def iter_of(cfg, name):
    cfg.loader.batch_size = 1                    # 冒烟降 batch,省 CPU
    print("配置 {}: arch={} seed={}".format(
        name, getattr(cfg.model, "model_name", type(cfg.model).__name__), cfg.train.seed))
    set_seeds(cfg.train.seed)
    dl, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=cfg.loader, dataset_config=cfg.data,
        world_size=None, rank=None, train_valid_test_kinds=["train"])
    return iter(dl["train"])


def make_iter(cfg_path):
    cfg = load_config(EXP, cfg_path)
    return cfg, iter_of(cfg, os.path.basename(cfg_path))


# --- 7a. swin:SI 训练前向 + 反向 3 iter ---
cfg_s, it_s = make_iter(CFG_DIR + "/config_wind_canvas_p3_arch_swin.yml")
net_s = make_model(cfg_s.model).to(DEV)
opt = torch.optim.AdamW(net_s.parameters(), lr=1e-5)
si = StochasticInterpolantFollmer(config=cfg_s.si, neural_net=net_s, device="cpu")
si.train()
for step in range(3):
    b = next(it_s)
    out = si(y0=b["y0"].to(DEV), y1=b["y"].to(DEV), y_cond=b["x"].to(DEV))
    opt.zero_grad()
    out.backward()
    opt.step()
    assert torch.isfinite(out).all(), "swin 损失非有限"
    print("  swin iter {} loss={:.6f}".format(step, float(out)))
print("SMOKE swin OK({:.3f} M 参数)".format(
    sum(p.numel() for p in net_s.parameters()) / 1e6))

# --- 7b. reg:回归步训练前向 + 反向 3 iter(与 optimize_reg 同式) ---
cfg_r, it_r = make_iter(CFG_DIR + "/config_wind_canvas_p3_arch_reg.yml")
reg = RegressionModel(cfg_r.model).to(DEV)
opt_r = torch.optim.AdamW(reg.parameters(), lr=1e-5)
reg.train()
for step in range(3):
    b = next(it_r)
    y0, y1, x = b["y0"].to(DEV), b["y"].to(DEV), b["x"].to(DEV)
    pred = reg.net(yt=y0, y_cond=x, gamma=torch.ones(y0.shape[0]))
    loss = F.l1_loss(pred, y1 - y0)
    opt_r.zero_grad()
    loss.backward()
    opt_r.step()
    assert torch.isfinite(loss).all(), "reg 损失非有限"
    print("  reg iter {} loss={:.6f}".format(step, float(loss)))
reg.eval()
with torch.no_grad():
    b = next(it_r)
    y1_full, aux = reg.sample_y1_bare_diffusion(y0=b["y0"].to(DEV), y_cond=b["x"].to(DEV))
assert aux is None and y1_full.shape == b["y"].shape and torch.isfinite(y1_full).all()
print("SMOKE reg OK(sample_y1_bare_diffusion 形状 {} 有限)".format(tuple(y1_full.shape)))

# --- 7c. edm:假回归(随机 UNet)+ 2 步 Heun 采样 1 次(可选;仅验证接线) ---
spec = importlib.util.spec_from_file_location(
    "train_edm_correction_smoke", "scripts/train_edm_correction.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
base_cfg, edm_cfg, reg_sec, _ = mod.load_edm_config(
    EXP, CFG_DIR + "/config_wind_canvas_p3_arch_edm.yml")
edm_cfg.sigma_data = 1.0                     # 未训练:占位 σ_d,只验形状/有限
cor = EDMCorrector(edm_cfg).to(DEV).eval()
it_e = iter_of(base_cfg, "config_wind_canvas_p3_arch_edm.yml(剥离 edm 段后)")
reg_fake = make_model(base_cfg.model).to(DEV).eval()   # 随机权重 = "假回归"
b = next(it_e)
pad16 = lambda t: F.pad(t, (0, 7, 0, 12), mode="replicate")   # (100,121) -> (112,128)
with torch.no_grad():
    y0p, xp = pad16(b["y0"].to(DEV)), pad16(b["x"].to(DEV))
    y1_reg = y0p + reg_fake(yt=y0p, y_cond=xp, gamma=torch.ones(y0p.shape[0]))
    out = cor.sample(y0=y0p, y1_reg=y1_reg, y_cond=xp, n_samples=1, steps=2, seed=0)
assert tuple(out.shape) == (1,) + tuple(y0p.shape) and torch.isfinite(out).all()
print("SMOKE edm OK(ctx={},样本形状 {},2 步 Heun)".format(
    edm_cfg.ctx_channel, tuple(out.shape)))
print("SMOKE ALL OK")
PY
echo "smoke exit=$?"

echo ""
echo "===== 8. 后续步骤 ====="
cat <<'EOM'
(1) W1 训练(swin + reg):
      bash ops/queue/64b_arch_waves.sh
(2) 查状态 / PEND 换队列:
      bash ops/queue/64c_check_arch.sh
(3) W1 训完 -> 评估(84;si 臂 N=16 集合均值 + 参考臂 _ens16 重评):
      bash ops/queue/64d_arch_eval.sh
(4) 闸门通过(前波评估产物存在)后再 W2:
      WAVE=2 bash ops/queue/64b_arch_waves.sh
(5) 终评报告(99 表 + ensemble json 汇总):
      bash ops/queue/64e_arch_report.sh
EOM
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result "configs/深圳/phase3_arch"
if [ -n "$(ls -A results/phase3_arch 2>/dev/null)" ]; then
  git add results/phase3_arch
fi
git commit -m "result 64 arch setup: configs + fixture/smoke gates"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
