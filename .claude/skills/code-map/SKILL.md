---
name: code-map
description: 项目代码地图与常见坑——src/ 四个子包的结构与关键 API、模型输入输出通道约定（in_channel=18+26=44）、config 加载机制与无 _path 属性的陷阱、checkpoint 路径规则、评估脚本族（evaluate_sz_experiments / evaluate_sz_A_group 的 modes）、绘图脚本与中文字体、数据切分的 myj/ysu 问题。当需要修改训练/评估/预处理代码、写新脚本、理解某段代码或排查报错时使用。
---

# 代码地图与常见坑

## 1. 目录结构

```
src/
├── dl_config/     # 配置加载:config_loader.load_config(experiment_name, path)
│                  #   YamlConfig.load() 递归转 dataclass —— 注意:返回对象无 _path 属性!
├── dl_data/       # dataloader.make_dataloaders_and_samplers(root_dir, loader_config,
│                  #   dataset_config, ..., train_valid_test_kinds=["test"])
│                  #   dataset_3d_wind.Dataset3dWind(继承 Dataset2dTemperature2m)
│                  #   __getitem__ 返回 {"x": 条件输入, "y": HR 目标, "y0": LR 风场}
├── dl_model/      # model_maker.make_model(config.model) —— 不要直接实例化类!
│                  #   ddpm/unet_ddpm_v01.UNetDDPMVer01
│                  #   si_follmer/si_follmer_framework.StochasticInterpolantFollmer
├── dl_utils/      # 随机种子等
scripts/           # 训练/评估/绘图/预处理脚本(见 §4)
configs/           # 模板/ 香港-基础/ 香港-全天消融/ 香港-昼夜消融-小,大模型/ 深圳/
lsf/               # 最终模型/ 昼夜消融/ ; ops/queue/ 任务脚本
```

## 2. 关键 API 与约定

**模型**
- `make_model(config.model)` → UNet;SI 框架:`StochasticInterpolantFollmer(config=config.si, neural_net=model)`
- 推理:`y_est, _ = si.sample_y1_bare_diffusion(y0=y0, y_cond=x)`
- UNet:4 次下采样(channel_mults 5 档)→ 输入边长须为 **16 的倍数**;`attn_res` 按 ds 倍数触发(96×112 下 [16] 正常,不用改)
- **输入张量 = cat([yt, y_cond]) → in_channel = 18(yt) + 26(全部输入) = 44**;输出 18(U/V/W × 6 eta 层)。HK 配置的 28 = 18+10 同理
- 物理约束:`config.si.divergence_weight / vorticity_weight`,施加在**漂移场 b**(模型输出)上;0 = 关闭

**数据**
- npz keys:`hr_u_ml0` 等 18 个目标、`lr_*` 18 个低精度风场、条件变量 8 个(t2,z,lu,tsk,hfx,lh,psfc,pblh)+ `lr_` 低精度版 + `swdown`(昼夜过滤用)
- 标准化:`config.data.biases / scales`,**每个数据集独立计算**,跨数据集勿混用
- 昼夜过滤:`config.data.day_night_filter = all/day/night`,dataloader 读 npz 的 swdown 均值过滤(白天>50,夜间<5 W/m²)
- **官方切分坑**:train/valid/test 按 `shuffle=False` 取排序文件段切分,文件名里 myj_ 全部排在 ysu_ 前 → **训练段含全部 myj+部分 ysu,测试段全为 ysu**。评估上做"每方案"分析须自行切分(见 evaluate_sz_A_group 的 scheme 模式)
- `loader.dataset.ps` = 按样本顺序的文件路径列表(test loader shuffle=False,顺序一致;用于按方案分组)

**Checkpoint**
- 路径:`data/DL_result/ExperimentSchrodingerBridge3dWind/<配置文件名去后缀>/checkpoint.pth`
- 如 `config_wind_3d_sz_baseline.yml` → `.../config_wind_3d_sz_baseline/checkpoint.pth`
- 加载:`torch.load(path, map_location="cpu", weights_only=False)`,取 `["model_state_dict"]`

## 3. 高频坑(真实踩过)

1. **`config._path` 不存在**——需要配置路径时显式传入,别从 config 对象取
2. **显示名 ≠ 技术名**:报告用 baseline/allLR/phys,**配置与 checkpoint 用 baseline/lrcond/pinn**;拼路径必须用技术名(映射见 sr-exp-design skill)
3. **直接实例化 UNetDDPMVer01 会缺 9 个参数**——必须走 make_model
4. 模型类名是 `StochasticInterpolantFollmer`,不是 SIFollmer;构造函数要 neural_net 参数
5. numpy 数组与 tensor 转换注意 device;评估函数 `compute_metrics(pred, target)` 接受 numpy [N,C,H,W]
6. 服务器脚本 Py2/3 兼容(见 server-ops skill)

## 4. 脚本族速查

| 脚本 | 用途 |
|------|------|
| `train_schrodinger_bridge_model.py` | 训练:`--config_path --experiment_name ExperimentSchrodingerBridge3dWind --device` |
| `evaluate_sz_experiments.py` | 深圳主评估;默认 5 模型(小+昼夜子集);`--d03` 跨域(第一轮模型在 d03)、`--d03train` 同域;指标函数 `compute_metrics` / `summarize_by_component` 是**全项目同口径基准** |
| `evaluate_sz_A_group.py` | A 组诊断:`--mode diag`(谱+逐层+分箱+分组)/`shuffle`(条件打乱)/`moments`(矩匹配)/`scheme`(每方案留出);模型参数用显示名(baseline/allLR/phys,内部自动映射) |
| `evaluate_ablation_day_night.py` | 香港昼夜消融评估(`--suffix _large`) |
| `evaluate_wind_3d.py` | 单模型评估 |
| `prepare_wind_data_3d_sz.py` / `_d03.py` | 深圳 d04 / d03 数据预处理(netCDF4, pytorch-gpu env 跑) |
| `inspect_wrfout.py` | WRF 文件探查(变量/网格/层数) |
| `plot_day_night_loss.py` / `plot_sz_report.py` / `plot_A1_spectra.py` | 绘图(matplotlib;中文字体 `PingFang SC` 等,已配) |
| `generate_day_night_ablation_configs.py` | 生成消融配置+LSF(`--large`) |

## 5. 改代码时的检查清单

- 新配置:按 `configs/深圳/` 现有模板改;`in_channel = 18 + len(input_variable_names)` + 18? **不**:in_channel = 18(yt) + 全部输入通道数(含 18 LR 风场),即 18+26=44
- 新脚本:import 路径用 `scripts/xxx.py` 里的既有函数(同口径);评估新实验必须与 `evaluate_sz_experiments.py` 同口径(同切分、同归一化、同指标函数)
- 改 dataloader/dataset:注意 HK 与深圳、d04 与 d03 都在用同一套类,改动要做兼容
- 训练配置的 epochs/学习率/物理权重:深圳 pinn 曾因固定权重后期震荡(最佳模型在中途),涉及时参考 sr-exp-design 的 C 组

## 6. 关联

- 实验设计约束:见 sr-exp-design skill;数据清单:见 sz-data-inventory;服务器操作:见 server-ops
