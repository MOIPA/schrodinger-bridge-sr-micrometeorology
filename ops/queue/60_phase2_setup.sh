#!/bin/bash
# 60_phase2_setup.sh — 阶段 2 前置门:数据/基线检查 -> 生成配置(86) -> 通道校验(81) -> 损失自检(95) -> 数据自检(80)
# 说明:86 生成的物理权重是占位值(86 的 WEIGHTS);60a 探针(96)出 results/phase2/probe_weights.json
#       后,按 suggested_weights 回填重跑 86,再重跑本脚本的第 2/3 步复核。
# 手机执行: bash ops/queue/60_phase2_setup.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
git pull --no-rebase
git --no-pager log --oneline -1
PY3D=/fsb/home/yutingwang/ytw_tangzq/.conda/envs/wind3d/bin/python
ROOT=/fsb/home/yutingwang/ytw_tangzq/schrodinger-bridge-sr-micrometeorology
EXP=ExperimentSchrodingerBridgeWindCanvas
OUT=ops/result/60_phase2_setup.txt
mkdir -p ops/result logs "configs/深圳/phase2" results/phase2
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
  echo "[缺] V* 基线 checkpoint: $VSTAR(60a 探针需要它)"
fi

echo ""
echo "===== 1. 生成阶段 2 配置(86;物理权重为占位值) ====="
$PY3D -u scripts/outline/86_gen_phase2_configs.py
echo "86 exit=$?"
ls -1 "configs/深圳/phase2" 2>/dev/null

echo ""
echo "===== 2. 通道校验(81;阶段 2 目录,含 phys_* 静态校验) ====="
$PY3D -u scripts/outline/81_config_channel_check.py --config_dir "configs/深圳/phase2"
echo "81 exit=$?"

echo ""
echo "===== 3. 物理损失项合成自检(95) ====="
$PY3D -u scripts/outline/95_phase2_loss_fixture.py
echo "95 exit=$?"

echo ""
echo "===== 4. canvas 数据集合成自检(80;含 rho 透传) ====="
$PY3D -u scripts/outline/80_fixture_check.py
echo "80 exit=$?"

echo ""
echo "===== 5. 后续步骤 ====="
cat <<'EOM'
(1) 探针定权重(约 10 分钟 GPU):
      bash ops/queue/60a_phase2_probe.sh
      -> results/phase2/probe_weights.json(suggested_weights;60a 会打印可直接执行的 86 命令)
(2) 用探针权重回填重生成配置(把 <v> 换成建议值):
      python scripts/outline/86_gen_phase2_configs.py --div-lo <v> --div-mid <v> --div-hi <v> \
        --spectral <v> --extreme <v> --vorticity <v>
    再重跑第 2/3 步复核:
      python scripts/outline/81_config_channel_check.py --config_dir "configs/深圳/phase2"
      python scripts/outline/95_phase2_loss_fixture.py
(3) W1 试点(2 个 run):
      bash ops/queue/60b_phase2_waves.sh 1      # W1 = p2_l2 p2_div_mid
(4) W1 训练完 -> 查状态 / 闸门评估 -> 人工判定:
      bash ops/queue/60c_check_phase2.sh
      bash ops/queue/60e_phase2_gate.sh         # -> results/phase2/ranking_phase2_gate.md
    判定通过后创建闸门文件并推送(本链路硬闸门,不自动创建):
      touch results/phase2/GATE_W1_PASS && git add results/phase2 && \
        git commit -m "gate: W1 pass" && git push
(5) W2(需 GATE_W1_PASS 已 pull 到服务器): bash ops/queue/60b_phase2_waves.sh 2
    W2 判定 -> GATE_W2_PASS -> W3: bash ops/queue/60b_phase2_waves.sh 3
(6) 全部训练完成后终评:
      bash ops/queue/60d_eval_phase2.sh
EOM
} > "$OUT" 2>&1

cat "$OUT"
git add ops/result
if [ -n "$(ls -A "configs/深圳/phase2" 2>/dev/null)" ]; then
  git add "configs/深圳/phase2"
fi
git commit -m "result 60 phase2 setup: configs + fixture gates"
git push || { git fetch origin && git merge --no-edit origin/main && git push; }
