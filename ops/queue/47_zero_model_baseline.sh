#!/bin/bash
# 47_zero_model_baseline.sh — 零模型(朴素插值)基线
# 直接用低分辨率输入当 1km 预测、不做任何学习,与 1km 真值比指标
# 回答"模型比不超分好多少"——超分领域标准的绝对参照
# d04:y0 = 模拟降采样+插值回 1km 的输入(相当于"只插值不超分")
# d03:y0 = 原生 3km 场重采样到 1km 网格(相当于"直接用 3km 场当 1km")
# 手机执行: bash ops/queue/47_zero_model_baseline.sh (纯数据计算,几分钟,不需要 GPU)
cd ~/schrodinger-bridge-sr-micrometeorology || exit 1
mkdir -p ops/result
OUT=ops/result/47_zero_model_baseline.txt
: > "$OUT"

echo "===== 零模型基线计算(与评估同口径:同 config、同 test 集、同指标函数) =====" >> "$OUT"
module load anaconda/3 2>/dev/null; source activate wind3d 2>/dev/null || conda activate wind3d 2>/dev/null
python - <<'PYEOF' >> "$OUT" 2>&1
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, ".")
ROOT = os.getcwd()
from src.dl_config.config_loader import load_config
from src.dl_data.dataloader import make_dataloaders_and_samplers
from scripts.evaluate_sz_experiments import compute_metrics, summarize_by_component

def zero_model(cfg_path, tag):
    config = load_config("ExperimentSchrodingerBridge3dWind", cfg_path)
    config.data.day_night_filter = "all"
    dict_loaders, _ = make_dataloaders_and_samplers(
        root_dir=ROOT, loader_config=config.loader,
        dataset_config=config.data, world_size=None, rank=None,
        train_valid_test_kinds=["test"])
    ds = dict_loaders["test"].dataset
    n = len(ds)
    target_names = config.data.target_variable_names

    y0s, y1s = [], []
    for i in range(n):
        s = ds[i]
        y0s.append(s["y0"])
        y1s.append(s["y"])
    y0 = torch.stack(y0s).float()
    y1 = torch.stack(y1s).float()

    # 与评估完全同口径:标准化空间直接算指标(报告 2.5 节口径)
    rmse, mae, ssim, corr, bias = compute_metrics(y0.numpy(), y1.numpy())
    m = {"rmse": rmse, "mae": mae, "ssim": ssim, "corr": corr, "bias": bias}
    r = summarize_by_component(m, target_names)
    print("== {}: n={}".format(tag, n))
    for k in ["Overall", "U", "V", "W"]:
        if k in r:
            v = r[k]
            print("  {:<8} RMSE {:.4f}  MAE {:.4f}  SSIM {:.4f}  Corr {:.4f}  bias {:.4f}".format(
                k, v["rmse"], v["mae"], v["ssim"], v["corr"], v["bias"]))
    return {"tag": tag, "n_samples": n, "by_component": r}

out = {}
out["d04_test"] = zero_model("configs/深圳/config_wind_3d_sz_baseline.yml", "d04 测试集(模拟低精度)")
out["d03_test"] = zero_model("configs/深圳/config_wind_3d_sz_d03_baseline.yml", "d03 测试集(真实低精度)")
with open("results/zero_model_baselines.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n已保存 results/zero_model_baselines.json")
PYEOF

echo "" >> "$OUT"
echo "===== 完成 =====" >> "$OUT"
cat "$OUT"

# 自动回传
cd ops && git add . && git commit -m "result 47_zero_model_baseline" && git push
