#!/bin/bash
# report4.sh — 外传-4《级联与迁移》结果表一键生成。
#
# 前置:三档重评(eval2_*)、交叉迁移重算(xfer2)、级联(27→9→3→1)都已产出。
# 缺哪个就打印 [警告] 并把该行留空,不会算出假数。
#
# 用法(登录节点即可,纯 numpy,不占 GPU):
#   bash sidequest_scale3x/queue/report4.sh
# 产物:
#   sidequest_scale3x/results/report4_tables.md   —— 可直接粘进报告的三张表
#   sidequest_scale3x/results/report4_tables.json —— 机器可读
set -u

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || { echo ">>> 仓库目录不存在: $REPO"; exit 2; }

PY3="${PY3:-python3}"
SQ=sidequest_scale3x/results
P4=results/phase4
OUT_MD="$SQ/report4_tables.md"
OUT_JSON="$SQ/report4_tables.json"
BLOCK="${BLOCK:-24}"
NBOOT="${NBOOT:-2000}"

# 固定 tag 顺序:参照(一步直连)在前,级联在后
REF="p4_27to1_reg"
TAGS="p4_27to1_reg,sq3to1_casc_casc,sq3to1_reg,sq9to3_casc_casc,sq9to3_reg,sq27to9_reg_casc,sq27to9_reg"

say() { echo "[report4] $*"; }

say "REPO=$REPO"
say "参照(一步 27→1 直连)= $REF"
say "tags = $TAGS"

mkdir -p "$SQ"

say "① 主表 + 逐日(配对块 bootstrap,块 ${BLOCK} h,${NBOOT} 次)"
"$PY3" sidequest_scale3x/scripts/cmp_casc.py \
    --dir "$SQ,$P4" \
    --ref "$REF" \
    --tags "$TAGS" \
    --block "$BLOCK" --n_boot "$NBOOT" \
    --per_day \
    --out_md "$OUT_MD" --out_json "$OUT_JSON" || say "  ⚠️ cmp_casc 失败(见上)"

say "② 单项汇总(每档 vs 自己的插值基线 y0)"
"$PY3" sidequest_scale3x/scripts/summarize3.py \
    --dir "$SQ" --md_out "$SQ/report4_summary.md" --json_out "$SQ/report4_summary.json" \
    || say "  ⚠️ summarize3(外传目录)失败"

"$PY3" sidequest_scale3x/scripts/summarize3.py \
    --dir "$P4" --md_out "$P4/summary4.md" --json_out "$P4/summary4.json" \
    || say "  ⚠️ summarize3(阶段4目录)失败"

say "③ 零样本迁移:增益保持率(retained = 迁移增益 / 自训增益)"
"$PY3" sidequest_scale3x/scripts/xfer_retention.py \
    --dir "$SQ" \
    --pairs "xfer9to3_by27to9_reg:sq9to3_reg,xfer27to9_by3to1_reg:sq27to9_reg" \
    --out_md "$SQ/xfer_retention.md" --out_json "$SQ/xfer_retention.json" \
    || say "  ⚠️ xfer_retention 失败"

for f in "$OUT_MD" "$SQ/report4_summary.md" "$P4/summary4.md" "$SQ/xfer_retention.md"; do
    if [ -s "$f" ]; then say "  产出 $f"; else say "  [缺] $f"; fi
done

say "完成。缺行说明对应产物还没跑到,不要手填。"
