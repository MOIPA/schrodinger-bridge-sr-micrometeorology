# -*- coding: utf-8 -*-
"""
生成深圳 d03 跨域评估配置(实验 #2:训练在模拟 LR、评估在原生 d03)。

归一化策略:
  - lr_*(d03 原生输入):用 d03 全量统计(1488 样本)——输入分布是 d03 的
  - hr_* / HR 条件(t2 等):保持 d04 统计(与训练时模型看到的分布一致)

生成两个配置:
  - config_wind_3d_sz_d03_baseline.yml:输入 18 lr wind(d03) + 8 HR 条件 —— baseline/pinn 模型用
  - config_wind_3d_sz_d03_lrcond.yml:输入 18 lr wind(d03) + 8 lr 条件(d03) —— lrcond 模型用(真实部署全低精度场景)

用法: python3 scripts/generate_sz_d03_eval_configs.py
"""
import copy
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).parent.parent
CONFIG_DIR = ROOT / "configs" / "深圳"

# ---- d03 全量统计(1488 样本,来自 compute_stats_d03.py 输出)----
# 只需 lr_* 26 个 key(18 风场 + 8 条件)
D03_BIASES_LR = {
    'lr_hfx': 47.959662, 'lr_lh': 103.720553, 'lr_lu': 13.238839,
    'lr_pblh': 415.888994, 'lr_psfc': 100150.633567, 'lr_t2': 302.360659,
    'lr_tsk': 303.653157, 'lr_z': 46.071095,
    'lr_u_ml0': 0.628650, 'lr_u_ml1': 0.812894, 'lr_u_ml10': 1.825027,
    'lr_u_ml2': 0.955556, 'lr_u_ml3': 1.082004, 'lr_u_ml5': 1.308994,
    'lr_v_ml0': 5.231603, 'lr_v_ml1': 6.113628, 'lr_v_ml10': 7.206476,
    'lr_v_ml2': 6.561234, 'lr_v_ml3': 6.840443, 'lr_v_ml5': 7.154155,
    'lr_w_ml0': -0.004646, 'lr_w_ml1': -0.007249, 'lr_w_ml10': -0.004422,
    'lr_w_ml2': -0.008597, 'lr_w_ml3': -0.008945, 'lr_w_ml5': -0.008309,
}
D03_SCALES_LR = {
    'lr_hfx': 99.982433, 'lr_lh': 104.398316, 'lr_lu': 4.827075,
    'lr_pblh': 263.531522, 'lr_psfc': 834.451610, 'lr_t2': 1.802459,
    'lr_tsk': 4.191284, 'lr_z': 66.733784,
    'lr_u_ml0': 3.126020, 'lr_u_ml1': 3.560501, 'lr_u_ml10': 4.730240,
    'lr_u_ml2': 3.787633, 'lr_u_ml3': 3.943826, 'lr_u_ml5': 4.182958,
    'lr_v_ml0': 2.909182, 'lr_v_ml1': 3.179654, 'lr_v_ml10': 3.630684,
    'lr_v_ml2': 3.333549, 'lr_v_ml3': 3.428533, 'lr_v_ml5': 3.543250,
    'lr_w_ml0': 0.070513, 'lr_w_ml1': 0.088527, 'lr_w_ml10': 0.117995,
    'lr_w_ml2': 0.098398, 'lr_w_ml3': 0.104053, 'lr_w_ml5': 0.111322,
}


def make_d03_config(src_cfg, input_names):
    """基于 d04 配置,换 dl_data_ver + 覆盖 lr_* 统计为 d03 值。"""
    cfg = copy.deepcopy(src_cfg)
    cfg['loader']['dl_data_ver'] = 'wrf_3d_v1_sz_d03'
    cfg['data']['input_variable_names'] = list(input_names)
    for k in list(cfg['data']['biases']):
        if k in D03_BIASES_LR:
            cfg['data']['biases'][k] = D03_BIASES_LR[k]
    for k in list(cfg['data']['scales']):
        if k in D03_SCALES_LR:
            cfg['data']['scales'][k] = D03_SCALES_LR[k]
    return cfg


def main():
    with open(CONFIG_DIR / "config_wind_3d_sz_baseline.yml") as f:
        base_cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "config_wind_3d_sz_lrcond.yml") as f:
        lrcond_cfg = yaml.safe_load(f)

    # 1) d03 baseline 评估配置:18 lr wind(d03) + 8 HR 条件
    c1 = make_d03_config(base_cfg, base_cfg['data']['input_variable_names'])
    out1 = CONFIG_DIR / "config_wind_3d_sz_d03_baseline.yml"
    with open(out1, "w") as f:
        yaml.safe_dump(c1, f, allow_unicode=True, sort_keys=False)
    print("生成:", out1)

    # 2) d03 lrcond 评估配置:18 lr wind(d03) + 8 lr 条件(d03)
    c2 = make_d03_config(base_cfg, lrcond_cfg['data']['input_variable_names'])
    out2 = CONFIG_DIR / "config_wind_3d_sz_d03_lrcond.yml"
    with open(out2, "w") as f:
        yaml.safe_dump(c2, f, allow_unicode=True, sort_keys=False)
    print("生成:", out2)

    # 一致性自检:所有输入 key 都应有统计
    for name, cfg in [("d03_baseline", c1), ("d03_lrcond", c2)]:
        missing_b = [k for k in cfg['data']['input_variable_names']
                     if k not in cfg['data']['biases']]
        missing_s = [k for k in cfg['data']['input_variable_names']
                     if k not in cfg['data']['scales']]
        assert not missing_b and not missing_s, \
            "{} 缺统计: {} {}".format(name, missing_b, missing_s)
        print("{} 输入 {} 通道,统计覆盖 OK".format(
            name, len(cfg['data']['input_variable_names'])))


if __name__ == "__main__":
    main()
