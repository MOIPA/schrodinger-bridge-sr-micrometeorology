#!/bin/bash
# 11_setup_datalink.sh — 建数据软链 + smoke 配置冒烟验证(配置加载/数据读取/模型前向)
# 手机执行: bash ops/queue/11_setup_datalink.sh
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result data/DL_data
OUT=ops/result/11_setup_datalink.txt
: > "$OUT"

echo "===== 1. 建数据软链 =====" >> "$OUT"
if [ -e data/DL_data/wrf_3d_v1_sz ] || [ -L data/DL_data/wrf_3d_v1_sz ]; then
  echo "(已存在)" >> "$OUT"
  ls -ld data/DL_data/wrf_3d_v1_sz >> "$OUT"
else
  ln -s ~/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d_sz data/DL_data/wrf_3d_v1_sz
  echo "(新建)" >> "$OUT"
  ls -ld data/DL_data/wrf_3d_v1_sz >> "$OUT"
fi
echo "软链内 npz 数:" >> "$OUT"
ls data/DL_data/wrf_3d_v1_sz/ 2>/dev/null | wc -l >> "$OUT"

echo "" >> "$OUT"
echo "===== 2. 冒烟验证(配置加载 + dataloader + 模型前向,CPU) =====" >> "$OUT"
module load anaconda/3 2>/dev/null || source /fs00/software/anaconda/3/etc/profile.d/conda.sh 2>/dev/null
source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python - "$OUT" >> "$OUT" 2>&1 <<'PYEOF'
import sys, os
out = sys.argv[1]
os.environ['PYTHONPATH'] = os.getcwd()
import torch
from src.dl_config.schrodinger_bridge_model_config import SchrodingerBridgeModelConfig
from src.dl_data.dataloader import make_dataloaders_and_samplers
from src.dl_model.si_follmer.si_follmer_framework import SIFollmerFramework

cfg = SchrodingerBridgeModelConfig.load('configs/深圳/config_wind_3d_sz_smoke.yml')
loaders, _ = make_dataloaders_and_samplers(
    root_dir=os.getcwd(), loader_config=cfg.loader,
    dataset_config=cfg.data, world_size=None, rank=None)
ds = loaders['train'].dataset
print('dataset size (train):', len(ds))
d0 = ds[0]
print('sample keys:', sorted(d0.keys()))
for k in sorted(d0.keys()):
    print('  ', k, tuple(d0[k].shape), d0[k].dtype)

# 复刻训练脚本的 batch 组装
y0 = torch.stack([ds[i]['y0'] for i in range(2)])
y1 = torch.stack([ds[i]['y'] for i in range(2)])
y_cond = torch.stack([ds[i]['x'] for i in range(2)])
print('y0', tuple(y0.shape), 'y1', tuple(y1.shape), 'y_cond', tuple(y_cond.shape))

si = SIFollmerFramework(cfg.si, cfg.model)
si = si.eval()
with torch.no_grad():
    loss = si(y0, y1, y_cond)
print('forward loss:', float(loss))
print('SMOKE OK')
PYEOF

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 11_setup_datalink" && git push
