# 3D风场超分辨率重建 — 技术报告

## 1. 任务概述

基于薛定谔桥（Schrödinger Bridge）框架，将WRF模式d04域的低分辨率三维风场重建为高分辨率三维风场。参考FuXi-CFD论文的思路，从2D风场超分（U10/V10，2通道）扩展为3D风场重建（6层×U/V/W=18通道）。

由于d01-d03数据已删除，仍使用降采样模拟低分辨率输入。

---

## 2. 数据说明

### 2.1 数据来源

- **WRF模式输出**：d04域（最高分辨率嵌套域）
- **文件路径**：`/public3/home/scg4074/gp_zhaokc/SimulationJuly/RESULT_UCM/SLUCM/wrfout_d04_*`
- **时间分辨率**：逐小时
- **空间网格**：180×180（质量点网格）

### 2.2 Arakawa C交错网格与去交错

WRF使用Arakawa C网格，不同变量存储在网格的不同位置：

```
        V(i,j+1)
         ↑
U(i,j) ← P(i,j) → U(i+1,j)
         ↓
        V(i,j)
```

- **标量（T, P等）**：存储在格子中心，维度为 `180×180`
- **U（东西风）**：存储在格子左右边界，维度为 `180×181`（x方向多1）
- **V（南北风）**：存储在格子上下边界，维度为 `181×180`（y方向多1）
- **W（垂直风）**：存储在格子顶底面，维度为 `(nz+1)×180×180`（z方向多1）

**去交错**：取相邻两点平均，将风场从边界位置插值回中心位置：

```python
U_center = (U[:, :, :-1] + U[:, :, 1:]) / 2    # 181→180，沿x
V_center = (V[:, :-1, :] + V[:, 1:, :]) / 2    # 181→180，沿y
W_center[k] = (W[k, :, :] + W[k+1, :, :]) / 2  # 沿z
```

### 2.3 垂直层级选取

选取6个eta模式层，近地面密、高层稀：

| LEVELS key | WRF bottom_top索引 | 大约高度 | 说明 |
|-----------|-------------------|---------|------|
| ml0 | 0 | ~10m | 最近地面层 |
| ml1 | 1 | ~30m | |
| ml2 | 2 | ~50m | |
| ml3 | 3 | ~70m | |
| ml5 | 5 | ~120m | |
| ml10 | 10 | ~300m | 边界层中上部 |

> **注意**：eta层是地形跟随坐标，实际高度随地形变化，以上高度为近似值。若需调整层级，修改 `prepare_wind_data_3d.py` 中的 `LEVELS` 字典即可。

### 2.4 目标变量（18通道）

模型需要超分重建的风场，6层 × 3分量 = 18个通道：

| 层级 | U（东西风） | V（南北风） | W（垂直风） |
|------|-----------|-----------|-----------|
| ml0 | hr_u_ml0 | hr_v_ml0 | hr_w_ml0 |
| ml1 | hr_u_ml1 | hr_v_ml1 | hr_w_ml1 |
| ml2 | hr_u_ml2 | hr_v_ml2 | hr_w_ml2 |
| ml3 | hr_u_ml3 | hr_v_ml3 | hr_w_ml3 |
| ml5 | hr_u_ml5 | hr_v_ml5 | hr_w_ml5 |
| ml10 | hr_u_ml10 | hr_v_ml10 | hr_w_ml10 |

每个HR变量都有对应的LR版本（`lr_u_ml0`, `lr_v_ml0`, ...），共36个风场数组。

### 2.5 条件/辅助变量（10通道）

提供地表物理环境信息，帮助模型理解风场形成的背景：

| WRF变量名 | npz key | 物理含义 | 作用 |
|----------|---------|---------|------|
| T2 | t2 | 2米气温 | 热力驱动风场 |
| HGT | z | 地形高度 | 地形直接影响风 |
| LU_INDEX | lu | 土地利用类型 | 城市/森林/水体摩擦力不同 |
| TSK | tsk | 地表温度 | 热岛效应、海陆风 |
| SWDOWN | swdown | 短波辐射（太阳） | 白天加热→对流→风 |
| GLW | glw | 长波辐射 | 夜间辐射冷却 |
| HFX | hfx | 感热通量 | 地表加热空气的强度 |
| LH | lh | 潜热通量 | 蒸发相关的能量交换 |
| PSFC | psfc | 地面气压 | 气压梯度驱动风 |
| PBLH | pblh | 边界层高度 | 湍流混合的范围 |

### 2.6 NPZ文件结构

每个时间步一个npz文件，共包含46个数组：

| 类别 | 数量 | 命名模式 | 原始shape |
|------|------|---------|-----------|
| HR风场 | 18 | `hr_{u,v,w}_ml{0,1,2,3,5,10}` | 180×180 |
| LR风场 | 18 | `lr_{u,v,w}_ml{0,1,2,3,5,10}` | 180×180 |
| 条件变量 | 10 | `t2, z, lu, tsk, swdown, glw, hfx, lh, psfc, pblh` | 180×180 |

---

## 3. 低分辨率数据生成方式

由于没有真实的低分辨率数据（d01-d03已删除），采用降采样模拟：

```
HR (180×180)
    ↓ F.avg_pool2d(kernel_size=4)    降采样，4×4窗口取平均
中间态 (45×45)                        模拟低分辨率
    ↓ F.interpolate(mode='bicubic')   双三次插值回原始尺寸
LR (180×180)                          存入npz
```

LR数据的空间分辨率等价于45×45，但像素数仍为180×180，与HR尺寸一致。这样模型输入输出shape相同，任务本质是**同尺寸的细节恢复/去模糊**。

---

## 4. 模型架构

### 4.1 UNet通道配置

```
模型输入 = cat(yt, y_cond) = 18 + 10 = 28 通道
    yt:     18通道（LR风场的插值状态）
    y_cond: 10通道（条件变量）

模型输出 = 18 通道（HR风场预测）
```

### 4.2 空间尺寸流程

```
npz中的原始数据 (180×180)
    ↓ F.interpolate resize
模型输入/输出 (192×192)    ← config中 hr_data_shape: [192, 192]
    ↓ 可选：插值回180×180
最终结果
```

选择192×192而非180×180的原因：UNet有5层下采样（channel_mults有5个元素），需要尺寸能被2⁵=32整除。192 = 32×6 ✓，且 192 > 180 不会丢失信息。

### 4.3 模型参数

| 参数 | 值 | 说明 |
|-----|-----|------|
| in_channel | 28 | 18(LR风) + 10(条件) |
| out_channel | 18 | 6层 × 3分量 |
| inner_channel | 64 | 从2D版的32提升 |
| res_blocks | 2 | 从2D版的1提升 |
| channel_mults | [1,2,4,8,8] | 5层下采样 |
| dropout | 0.2 | |
| attn_res | [16] | 在16×16分辨率加attention |

### 4.4 训练参数

| 参数 | 值 |
|-----|-----|
| learning_rate | 0.0005 |
| batch_size | 2 |
| epochs | 400 |
| optimizer | AdamW |
| loss | L2 |
| use_amp | false（CPU训练） |
| SI n_timestep | 10 |
| SI formula | quadratic |

---

## 5. 代码文件说明

### 5.1 新建文件

| 文件 | 说明 |
|------|------|
| `prepare_wind_data_3d.py` | 数据准备：从WRF提取3D风场，去交错，生成HR/LR对，计算bias/scale |
| `src/dl_data/dataset_3d_wind.py` | Dataset子类：继承2D版，覆盖`_get_var_name`使用完整变量名 |
| `configs/config_wind_3d.yml` | 3D风场配置文件 |
| `visualize_wind_3d.py` | 可视化：水平风场对比、W分量对比、垂直剖面图 |

### 5.2 修改的文件

| 文件 | 改动 |
|------|------|
| `src/dl_data/dataloader.py` | 添加Dataset3dWind导入和elif分支 |
| `src/dl_config/schrodinger_bridge_model_config.py` | 添加ExperimentSchrodingerBridge3dWindConfig |
| `src/dl_config/config_loader.py` | 添加3D Wind实验的分发 |
| `scripts/train_schrodinger_bridge_model.py` | 添加`--experiment_name`命令行参数 |

### 5.3 未修改的文件

| 文件 | 原因 |
|------|------|
| `src/dl_model/si_follmer/si_follmer_framework.py` | 已经是通道无关的 |
| `src/dl_model/ddpm/unet_ddpm_v01.py` | in/out_channel由配置驱动 |
| `src/dl_train/si_optim_helper.py` | 只传batch字典，不关心通道数 |

---

## 6. 运行步骤

### 步骤1：准备数据
```bash
cd /public3/home/scg4074/tzq/git/schrodinger-bridge-sr-micrometeorology
python prepare_wind_data_3d.py
```
交互输入处理天数，完成后将打印的biases和scales复制到 `configs/config_wind_3d.yml`。

### 步骤2：创建数据目录软链接
```bash
mkdir -p data/DL_data
ln -s /public3/home/scg4074/tzq/git/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d data/DL_data/wrf_3d_v1
```

### 步骤3：数据验证（可选）
```bash
python visualize_wind_3d.py --npz_dir data/DL_data/wrf_3d_v1 --data_only --num_samples 1
```

### 步骤4：冒烟测试
```bash
python scripts/train_schrodinger_bridge_model.py \
  --config_path configs/config_wind_3d.yml \
  --experiment_name ExperimentSchrodingerBridge3dWind \
  --device cpu
```

### 步骤5：可视化结果
```bash
python visualize_wind_3d.py \
  --npz_dir data/DL_data/wrf_3d_v1 \
  --config_path configs/config_wind_3d.yml \
  --model_weight_path data/DL_result/ExperimentSchrodingerBridge3dWind/config_wind_3d/checkpoint.pth
```

---

## 7. 显存/内存估算

- 18通道输出 + inner_channel=64 + 192×192 → 参数量约20M+
- CPU训练：内存需求约8-16GB（batch_size=2）
- 如果内存不够：降`hr_data_shape`到`[160, 160]`或降`inner_channel`到48

---

## 8. 与2D风场版的对比

| | 2D风场版 | 3D风场版 |
|---|---------|---------|
| 目标变量 | U10, V10 (2通道) | 6层×U/V/W (18通道) |
| 条件变量 | 24个(含模式层变量) | 10个(仅地表) |
| UNet输入 | 26通道 | 28通道 |
| UNet输出 | 2通道 | 18通道 |
| inner_channel | 32 | 64 |
| res_blocks | 1 | 2 |
| hr_data_shape | 128×128 | 192×192 |
| 变量名截断 | name[:5] | 完整变量名 |
| 实验名 | ExperimentSchrodingerBridgeModel | ExperimentSchrodingerBridge3dWind |
