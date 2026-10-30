"""arch_params.py — 只建模型、不碰数据:核对各判据臂的参数量与配置可达性。

用途:在提交 GPU 训练前,用 CPU 几十秒验证
  ①配置能被 load_config 解析(字段名/嵌套没错)
  ②make_model 能按 arch 建出网络(swinir_canvas / unet_ddpm_v01 分派对)
  ③参数量与文档声称的一致(UNet 79.26 M / SwinIR 19.2 M / UNet-small 19.87 M)

在服务器登录节点:
    cd ~/schrodinger-bridge-sr-micrometeorology
    module load anaconda/3 && source activate wind3d
    python ops/debug/arch_params.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_model.model_maker import make_model  # noqa: E402

EXP = "ExperimentSchrodingerBridgeWindCanvas"
CFG_DIR = "configs/深圳/phase3_arch"
TAGS = [
    "p3_arch_reg",
    "p3_arch_reg_s2",
    "p3_arch_unet_small",
    "p3_arch_swin",
    "p3_arch_edm",
]


def main() -> int:
    bad = 0
    print(f"{'tag':26s} {'model_name':16s} {'params':>10s}  {'in':>4s} {'out':>4s}  status")
    for t in TAGS:
        path = os.path.join(CFG_DIR, f"config_wind_canvas_{t}.yml")
        try:
            cfg = load_config(EXP, path)
            model = make_model(cfg.model)
            n = sum(p.numel() for p in model.parameters()) / 1e6
            print(
                f"{t:26s} {cfg.model.model_name:16s} {n:9.2f}M  "
                f"{cfg.model.in_channel:4d} {cfg.model.out_channel:4d}  OK"
            )
            del model
        except Exception as e:  # noqa: BLE001 - 目的就是把任何异常显示出来
            bad += 1
            print(f"{t:26s} {'-':16s} {'-':>10s}  {'-':>4s} {'-':>4s}  FAIL: {type(e).__name__}: {e}")
    print(f"\n失败 {bad} / {len(TAGS)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
