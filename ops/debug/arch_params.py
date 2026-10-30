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
import traceback

sys.path.insert(0, os.getcwd())

from src.dl_config.config_loader import load_config  # noqa: E402
from src.dl_model.edm_correction import EDMCorrector  # noqa: E402
from src.dl_model.model_maker import make_model  # noqa: E402
from scripts.train_edm_correction import load_edm_config  # noqa: E402

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
            if t.startswith("p3_arch_edm"):
                # edm 臂不带 edm/reg 段走统一 loader:配置里 edm/reg 两段必须由
                # load_edm_config 剥离后进 base loader,直接 load_config 会 KeyError
                base_cfg, edm_cfg, reg_raw, _ = load_edm_config(EXP, path)
                if not edm_cfg.sigma_data or float(edm_cfg.sigma_data) <= 0:
                    import dataclasses as _dc
                    edm_cfg = _dc.replace(edm_cfg, sigma_data=1.0)
                model = EDMCorrector(edm_cfg)
                n = sum(p.numel() for p in model.parameters()) / 1e6
                print(
                    f"{t:26s} {'edm_correction':16s} {n:9.2f}M  "
                    f"{base_cfg.model.in_channel:4d} {base_cfg.model.out_channel:4d}  OK"
                    f"  (reg 段: {reg_raw.get('reg_tag', '-')})"
                )
                del model
                continue
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
            # 完整栈:配置类问题(如 KeyError: 'edm' = base loader 收到未剥离的 edm 段)
            # 只看类名+消息不足以定位,这里把栈打到 stderr
            print(traceback.format_exc(), file=sys.stderr)
    print(f"\n失败 {bad} / {len(TAGS)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
