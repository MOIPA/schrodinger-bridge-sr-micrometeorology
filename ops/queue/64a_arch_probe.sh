#!/bin/bash
# 64a_arch_probe.sh — 架构计时探针(GPU 作业;登录节点无 GPU):
#   真实 train loader 上实测单 iter 前向+反向耗时与显存峰值:
#   UNet(inner 64,参考臂) / UNet(inner 32,容量对齐臂) / SwinIR(inner 288/536)
#   用途:决定 SwinIR 最终容量(536≈19.2M 与 UNet-small 19.87M 对齐;若耗时超预算
#   则降 inner 并在报告标注容量差)。结果写 ops/result/64a_arch_probe.txt。
# 手机/登录节点执行: bash ops/queue/64a_arch_probe.sh   (提交后立即返回)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
QUEUES="e5v4p100ib 6148v100ib 7552v100 62v100ib 83a100ib"

if [ "$1" != "--inner" ]; then
  git pull --no-rebase
  git --no-pager log --oneline -1
  mkdir -p ops/result logs results/phase3_arch
  Q=""
  for q in $QUEUES; do
    PEND=$(bqueues -w "$q" 2>/dev/null | tail -1 | awk '{print $9}')
    if [ "$PEND" = "0" ] 2>/dev/null; then Q="$q"; break; fi
  done
  [ -z "$Q" ] && Q="83a100ib"
  GITDIR=$(dirname "$(command -v git)")
  echo "提交架构计时探针(约 10 分钟)-> 队列 $Q"
  bsub -q "$Q" -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
    -J "p3archprobe" -o "logs/p3archprobe_%J.out" -e "logs/p3archprobe_%J.err" \
    "cd $ROOT && export PATH=$GITDIR:\$PATH && bash ops/queue/64a_arch_probe.sh --inner"
  exit 0
fi

# ---------------- GPU 节点内 ----------------
cd "$ROOT" || exit 1
OUT=ops/result/64a_arch_probe.txt
: > "$OUT"
$PY3D -u - <<'PY' >> "$OUT" 2>&1
import copy
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.model_maker import make_model
from src.dl_model.swinir_arch import SwinIRCanvasConfig

EXP = "ExperimentSchrodingerBridgeWindCanvas"
cfg = load_config(EXP, "configs/深圳/phase2/config_wind_canvas_p2_l1r2_lr2e4.yml")
dev = torch.device("cuda:0")
dl, _ = make_dataloaders_and_samplers(
    root_dir=os.getcwd(), loader_config=cfg.loader, dataset_config=cfg.data,
    world_size=None, rank=None, train_valid_test_kinds=["train"])
it = iter(dl["train"])
batches = []
for _ in range(3):
    b = next(it)
    batches.append((b["y0"].to(dev), b["y"].to(dev), b["x"].to(dev)))
del it, dl
print("batch: y0 %s, x %s" % (tuple(batches[0][0].shape), tuple(batches[0][2].shape)))


def time_net(build, name, n=5):
    net = build().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    y0, y, x = batches[0]
    net.train()
    t0 = time.time()
    for i in range(n):
        y0, y, x = batches[i % len(batches)]
        opt.zero_grad(set_to_none=True)
        out = net(yt=y0, y_cond=x, gamma=torch.ones(y0.shape[0], device=dev))
        loss = (out - (y - y0)).abs().mean()
        loss.backward()
        opt.step()
    dt = (time.time() - t0) / float(n)
    mem = torch.cuda.max_memory_allocated(dev) / (2 ** 30)
    torch.cuda.reset_peak_memory_stats(dev)
    nparam = sum(p.numel() for p in net.parameters()) / 1e6
    print("%-26s %.1f s/it  peak-alloc %.2f GiB  %.2fM params" % (name, dt, mem, nparam))


m = cfg.model
time_net(lambda: make_model(m), "UNet inner=%d(ref)" % m.inner_channel)
m32 = copy.deepcopy(m)
m32.inner_channel = 32
time_net(lambda: make_model(m32), "UNet inner=32")
for inner in (288, 536):
    sc = SwinIRCanvasConfig(in_channel=167, out_channel=72, inner_channel=inner,
                            num_blocks=6, window_size=8, num_heads=4, mlp_ratio=2.0,
                            dropout=0.0)
    time_net(lambda sc=sc: make_model(sc), "SwinIR inner=%d" % inner)
print("PROBE OK")
PY
echo "exit=$?" >> "$OUT"

cat "$OUT"
git add ops/result
git commit -m "result 64a arch timing probe"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
