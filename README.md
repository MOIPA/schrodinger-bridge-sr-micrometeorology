# 基于薛定谔桥的城市微气象概率超分辨率

## 1. 项目目标

本项目旨在对论文《Probabilistic Super-Resolution for Urban Micrometeorology via a Schrödinger Bridge》中提出的方法进行复现和改造。

核心目标是利用**薛定谔桥（Schrödinger Bridge, SM）**模型，对气象数据（特别是2米气温）进行超分辨率重建。原始代码库经过了大量修改，以支持在本地、仅有CPU的Windows环境下运行，并使用公开的**ERA5再分析数据**替代了原始的私有数据集。

### 核心方法论
本项目的核心思想是将“超分辨率”问题构建为一个**输运问题（Transport Problem）**，而不是一个简单的、确定性的图像映射。它将超分过程建模为从**低分辨率（LR）图像分布**到**高分辨率（HR）图像分布**的一条概率性的“演化路径”。

薛定谔桥（Schrödinger Bridge）是用于寻找这条演化路径最高效解的数学框架。在代码实现中，我们使用了一个 **U-Net** 架构作为“引擎”，来近似求解描述这一演化过程的随机微分方程（SDE）中的动态项（漂移项）。

---

## 2. 工作流程与分步指南

本节提供了从数据获取到最终可视化的完整项目操作流程。

### 第1步：数据获取

模型使用ERA5再分析数据进行训练。您需要下载“单层”和“等压层”两类数据。

*   **数据源**: 哥白尼气候数据中心 (Copernicus Climate Data Store, CDS)
*   **下载链接**:
    *   单层数据: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)
    *   等压层数据: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels)
*   **所需变量**:
    *   **从“单层数据”下载**:
        *   `2m temperature` (2米气温)
        *   `10m u-component of wind` (10米U分量风)
        *   `10m v-component of wind` (10米V分量风)
        *   `Geopotential` (位势，即地形高度)
        *   `Land-sea mask` (海陆掩码)
    *   **从“等压层数据”下载**:
        *   `Temperature` (温度 t)
        *   `U-component of wind` (U分量风 u)
        *   `V-component of wind` (V分量风 v)
        *   `Vertical velocity` (垂直速度 w)
        *   **等压层 (Pressure Levels)**: 850, 875, 900 hPa

请将数据下载为 `GRIB` 格式。

### 第2步：数据预处理

下载原始的 `.grib` 文件后，需要运行脚本将其转换为模型所需的 `.npz` 格式。

*   **脚本**: `prepare_data.py`
*   **输入**: 放置在项目根目录下的原始 `.grib` 文件 (您也可以在脚本中修改路径)。
*   **功能**: 该脚本会读取GRIB文件，选择一个特定的地理区域和时间范围（可在脚本内修改），生成对应的低分辨率/高分辨率数据对，并保存为带时间戳的独立 `.npz` 文件。
*   **输出位置**: `data/DL_data/v20_2d/`
*   **输出格式**: 每个 `.npz` 文件都是一个包含多个Numpy数组的字典，键名对应第一步中下载的各个变量，例如 `hr_tm002m`, `lr_tm002m`, `z`, `lsm`, `t_900` 等。

### 第3步：参数配置

所有主要的训练和数据参数都由一个中央配置文件控制。

*   **文件**: `configs/schrodinger_bridge_model.yml`
*   **关键参数**: 在训练前，您可能需要检查此文件中的 `batch_size` (批大小), `hr_data_shape` (数据尺寸), `hr_cropped_shape` (裁剪尺寸), 以及 `input_variable_names` (输入变量列表) 等，确保它们与您的数据集和目标一致。

### 第4步：模型训练

训练脚本会启动训练过程，并已集成“断点续训”功能。

*   **脚本**: `scripts/train_schrodinger_bridge_model.py`
*   **运行命令**:
    ```bash
    python scripts/train_schrodinger_bridge_model.py --config_path configs/schrodinger_bridge_model.yml --device cpu
    ```
*   **断点续训**:
    *   第一次运行时，脚本会从头开始训练。
    *   当模型效果提升时，一个 `checkpoint.pth` 文件会被保存在结果目录中 (例如 `data/DL_result/...`)。
    *   如果您中途停止并重新运行训练，脚本会自动找到此文件，并从上次的进度（包括模型权重、优化器状态和轮次）继续训练。

### 第5步：推理与可视化

训练完成后，您可以使用可视化脚本对单个数据文件进行推理并查看结果。

*   **脚本**: `visualize_samples.py`
*   **运行命令**:
    ```bash
    python visualize_samples.py ^
        --config_path configs/schrodinger_bridge_model.yml ^
        --model_weight_path path/to/your/checkpoint.pth ^
        --npz_path path/to/your/data.npz ^
        --output_path map_comparison_final.png
    ```
    *   **注意**: 请将 `path/to/your/checkpoint.pth` 替换为您实际保存的断点文件路径，并将 `path/to/your/data.npz` 替换为您想要测试的任意一个 `.npz` 文件的路径。
*   **输出**: 脚本会生成一张PNG图片，图中包含低分辨率输入、高分辨率真实值、以及模型的高分辨率预测结果，所有图像都绘制在真实的地理地图背景上。



### 评估


⏺ 评估脚本和SLURM提交脚本都写好了。     
                                                                                                                                  
  生成了两个文件
                                                                                                                                                                                                  
  scripts/evaluate_wind_3d.py — 评估脚本，功能：
  1. 加载训练好的模型，在test集上跑推理                                                                                                                                                           
  2. 反标准化回物理单位（m/s）                        
  3. 同时计算bicubic baseline（直接用LR数据当预测，不需要额外训练）
  4. 逐通道算 RMSE / MAE / SSIM / Correlation                      
  5. 按风分量(U/V/W) 和层级(ml0~ml10) 分组汇总
  6. 输出到终端 + 保存 metrics.json + metrics.csv                                                                                                                                                 
   
  task-eval-wind-3d.sh — SLURM提交脚本                                                                                                                                                            
                                                      
  在超算上运行

  # 1. git同步后
  cd /public3/home/scg4074/tzq/git/schrodinger-bridge-sr-micrometeorology

  # 2. 检查行尾（防止dos换行问题）
  sed -i 's/\r$//' task-eval-wind-3d.sh

  # 3. 提交
  sbatch task-eval-wind-3d.sh

  输出示例大概长这样：
  Variable       | Model RMSE  Bicubic   | Model MAE   Bicubic   | Model SSIM  Bicubic   | Model Corr  Bicubic
  hr_u_ml0       |      0.8234    1.2567 |      0.6123    0.9845 |      0.9312    0.8567 |      0.9856    0.9523
  hr_v_ml0       |      0.7891    1.1234 |      ...
  hr_w_ml0       |      0.0345    0.0678 |      ...
  ...
  分组汇总:
    Component U     : RMSE 1.23±0.34 (bicubic 1.89±0.45) | SSIM 0.92±0.03 ...
    Component W     : RMSE 0.04±0.01 (bicubic 0.07±0.02) | ...
    Level ml0       : ...

  这样就有了模型 vs bicubic 的逐通道对比表，可以直接放PPT和论文。




⏺ 全部验证通过。清理多余的pending任务：

⏺ 全部完成！总结一下生成的文件：

  代码改动（1个文件）

  src/dl_model/si_follmer/si_follmer_framework.py — 3处改动：
  - SIFollmerConfig 添加 channel_weights 可选字段
  - __init__ 解析权重为 [1,C,1,1] 的buffer
  - forward 的L2 loss乘以通道权重

  旧配置（没有channel_weights）完全不受影响，走 None 分支。

  新建文件（13个）

  ┌─────────────────────────────────────────────────────┬────────────────────────────────────────┐
  │                        文件                         │                  用途                  │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ W加权实验                                           │                                        │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ configs/config_wind_3d_wloss.yml                    │ W权重=10，U/V=1                        │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ task-wind-3d-wloss.sh                               │ SLURM提交                              │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ 消融实验（5组）                                     │                                        │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ configs/config_wind_3d_ablation_all.yml             │ 对照组（全10个条件变量，in_ch=28）     │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ configs/config_wind_3d_ablation_no_terrain.yml      │ 去HGT+LU（in_ch=26）                   │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ configs/config_wind_3d_ablation_no_thermal.yml      │ 去T2/TSK/SWDOWN/GLW/HFX/LH（in_ch=22） │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ configs/config_wind_3d_ablation_no_pblh.yml         │ 去PBLH（in_ch=27）                     │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ configs/config_wind_3d_ablation_no_pressure.yml     │ 去PSFC（in_ch=27）                     │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ task-ablation-all.sh ~ task-ablation-no-pressure.sh │ 5个SLURM脚本                           │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ 批量提交                                            │                                        │
  ├─────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ submit_all_experiments.sh                           │ 一键提交全部6个实验                    │
  └─────────────────────────────────────────────────────┴────────────────────────────────────────┘

  在超算上运行

  cd /public3/home/scg4074/tzq/git/schrodinger-bridge-sr-micrometeorology

  # 修复行尾
  sed -i 's/\r$//' task-*.sh submit_all_experiments.sh

  # 一键提交全部6个实验
  chmod +x submit_all_experiments.sh
  ./submit_all_experiments.sh

  # 查看队列
  squeue -u $USER

  6个实验会并行排队，每个约22小时。训练完后用 evaluate_wind_3d.py 对每组模型跑评估，就能出消融对比表了。
