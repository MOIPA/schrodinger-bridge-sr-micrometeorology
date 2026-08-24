# 新 Agent 交接指南

> 本文件是给接手本项目的 Claude Code agent 的完整交接文档（2026-08-25 建立）。
> 在另一台电脑上工作的新 agent 没有任何历史会话记忆，所有必要知识都在本文件
> 和 `docs/项目历史与数据说明.md`（数据演化史、路径表、目录指南）中。
> 先读两份文档，再动任何东西。

---

## 0. 项目一句话

**薛定谔桥（Stochastic Interpolant 扩散框架）3D 风场超分辨率重建：WRF 3km → 1km，
模型输出"漂移场 b"逐步修正低分辨率风场到高分辨率，带物理约束（散度 + 大尺度涡度）。**

数据演进：ERA5 2D 气温（已废弃）→ 香港 WRF 3D 风场（当前核心，全部实验完成）
→ 深圳 WRF（进行中）。

---

## 1. 当前状态与下一步（最重要）

### 正在进行的：深圳数据预处理

- 服务器登录节点后台进程（nohup，PID 26565），2026-08-24 22:06 启动，约 45 分钟，**现在应已完成**
- 脚本：`scripts/prepare_wind_data_3d_sz.py`，日志：服务器 `logs/prep_sz_nohup.out`
- 输出：服务器 `~/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d_sz/`（**8928 个 npz**）
- 预期统计：myj 4464 + ysu 4464；昼夜各约 4000+（7 月太阳几何划分）

### 验证步骤（用户重连服务器后第一步）

```bash
# 1. 数量
ls prepare_npz_wind_3d_sz/ | wc -l                    # 应为 8928
ls prepare_npz_wind_3d_sz/ | sed 's/_.*//' | sort | uniq -c   # myj 4464 + ysu 4464
# 2. 日志尾部（biases/scales 统计 + 昼夜样本数）
tail -40 logs/prep_sz_nohup.out
# 3. 抽查一个 npz 的 key 和 shape（应为 45 个 key，全部 (96,112)）
#    key: 18 个 hr_* + 18 个 lr_* + t2/z/lu/tsk/hfx/lh/psfc/pblh + swdown(合成)
```

### 下一步路线图

| 步骤 | 做什么 | 关键点 |
|------|--------|--------|
| 1 | 验证预处理结果 | 上表 |
| 2 | 建数据软链 | `ln -s ~/.../prepare_npz_wind_3d_sz data/DL_data/wrf_3d_v1_sz`（训练读取 `data/DL_data/<dl_data_ver>`） |
| 3 | 写深圳配置 | `configs/深圳/`（模板见下） |
| 4 | 训练 | LSF 提交（见 §6） |
| 5 | 评估 | `scripts/evaluate_ablation_day_night.py` 类似流程 |
| 6 | **NS 全约束实验** | 深圳数据支持完整动量方程（见 §5）——这是本次适配的核心目标 |

### 深圳配置模板（在 `configs/深圳/` 新建）

复制 `configs/香港-昼夜消融-小模型/config_wind_3d_ablation_all_day.yml` 修改：

- `model.in_channel: 26`（18 LR + 8 条件：t2, z, lu, tsk, hfx, lh, psfc, pblh；**无 swdown/glw**）
- `model.out_channel: 18`（不变）
- `data.hr_data_shape: [96, 112]`、`data.hr_cropped_shape: [96, 112]`（深圳域，非方形！）
- `data.input_variable_names`：t2, z, lu, tsk, hfx, lh, psfc, pblh（8 个，去掉 swdown/glw）
- `data.biases / data.scales`：用预处理日志打印的深圳统计量（**必须替换，香港的不能用**）
- `loader.dl_data_ver: wrf_3d_v1_sz`（配合软链）
- `data.day_night_filter`：all / day / night 均可（swdown 为合成值，过滤逻辑不变）
- `si.divergence_weight / vorticity_weight`：默认 0（消融纯 L2）；NS 实验再加
- 注意：模型 4 次下采样要求边长 16 的倍数，96 和 112 均满足

---

## 2. 服务器访问（关键操作手册）

### SSH 双层密码，agent 无法自己登录

第一层密码固定、第二层动态 OTP，**由用户输入**。工作流：

1. 用户在新终端执行：`screen -S srv` → `ssh 用户名@服务器` → 输两层密码 → `cd ~/schrodinger-bridge-sr-micrometeorology` → `git pull` → `Ctrl-A` `D` 脱离
2. agent 侧通过 screen 驱动（Mac 自带 screen）：
   - 发送命令：`screen -S srv -X stuff $'命令\n'`
   - 读输出：`screen -S srv -X hardcopy /tmp/x.txt; tail -30 /tmp/x.txt`
   - 会话名可能带 PID 前缀（如 `60030.srv`），用 `screen -ls` 查
3. **SSH 会掉线**：掉线后 screen 里变成本地 shell，让用户重新 ssh 即可；nohup 任务不受影响

### 服务器环境

| 项目 | 值 |
|------|-----|
| 服务器 | login1，用户名 ytw_tangzq，主机 entry.nju.edu.cn |
| 项目路径 | `~/schrodinger-bridge-sr-micrometeorology/`（git 同步） |
| 训练 env | `wind3d`（**无 netCDF4**，训练/评估用） |
| 探查/预处理 env | `pytorch-gpu`（**有 netCDF4**）：`/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python` |
| 登录节点默认 python | **2.7**（无 netCDF4，f-string 会挂） |

**脚本规则：必须 Python 2/3 兼容（无 f-string、纯 ASCII）**——除非明确知道只会用 py3 env 跑。

### LSF 提交

```bash
bsub -q 723090ib -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" -J job名 \
  -o logs/job名_%J.out "cd ~/schrodinger-bridge-sr-micrometeorology && python scripts/xxx.py ..."
```

- **铁律：GPU 队列（723090ib）会杀不用 GPU 的任务（约 4 分钟 SIGTERM）**。
  训练/评估带 `-gpu` 且真的用 GPU → 没事；**纯 CPU 数据任务（预处理等）不要 bsub，在登录节点 nohup 跑**
- 提交前 `module load anaconda/3 && source activate wind3d`

### GitHub 同步

- 远程：`github.com/MOIPA/schrodinger-bridge-sr-micrometeorology`（main 分支）
- **push 失败（连接超时）时加本地代理**：`http_proxy=http://127.0.0.1:7892 https_proxy=http://127.0.0.1:7892 git push`（换电脑后代理地址可能变，问用户）
- 用户负责本地 push，服务器 `git pull` 拉取；agent 也可直接 commit+push（用户已授权）

### 数据磁盘

- 服务器 /fsb 有 408T 空间；数据都在服务器上，本地只是副本
- `data/`、`logs/`、`prepare_npz_*` 均在 .gitignore（不会进 git）

---

## 3. 数据档案

### 香港（全部实验完成，用于对比/复现）

| 项 | 值 |
|----|-----|
| npz 位置（服务器） | `~/schrodinger-bridge-sr-micrometeorology/prepare_npz_wind_3d/` |
| 训练读取路径 | `data/DL_data/wrf_3d_v1`（指向上述目录） |
| 样本 | 480 个，2018-07，128×128，float32 |
| 原始 WRF | 旧服务器 `/public3/...`（本机不可访问，不用管） |
| 6 层 | eta 索引 0,1,2,3,5,10（≈30m–10km） |
| 条件变量 | t2, z, lu, tsk, swdown, glw, hfx, lh, psfc, pblh（10 个全用） |

### 深圳（进行中）

| 项 | 值 |
|----|-----|
| 原始 WRF | `/fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj/` 和 `meso_202007_ysu/` |
| 网格 | d04 = 1km，质量网格 99×120，**61 层**，10-min 输出，每文件 6 时次，744 文件/方案 |
| 预处理后 | `prepare_npz_wind_3d_sz/`，**8928 样本**（myj+ysu），96×112，文件名 `myj_20200701T000000.npz` |
| 预处理决策 | 去 5 点缓冲区→89×110→微 pad 96×112；eta 索引与香港一致；**无辐射变量→太阳几何合成 swdown（仅过滤用）** |
| 条件变量 | t2, z, lu, tsk, hfx, lh, psfc, pblh（8 个，无 swdown/glw） |
| 其他数据 | les_20200713、les_20200730（111m/37m LES，还没用） |

### 服务器上的结果/检查点

- 训练检查点：`data/DL_result/ExperimentSchrodingerBridge3dWind/<配置名>/checkpoint.pth`
  （配置名 = yml 文件名去后缀，如 `config_wind_3d_ablation_all_day`）
- 评估 JSON：`results/ablation_day_night/*_metrics.json` + `all_summaries.json`（含小+大模型）
- loss 图：`results/ablation_day_night/loss_*.png`（日志已从 git 删除，原始日志在服务器 `logs/`）

---

## 4. 模型与训练细节

### I/O

- 输入：18 个 LR 风场（hr 的 4 倍平均池化再双三次插回，即"降采样再还原"）+ 条件变量
  - 香港 All10：28ch；NoPBLH 27 / NoTerrain 26 / NoThermal 22 / NoPSFC 27
  - 深圳：26ch
- 输出：18 个 HR 风场（U/V/W × 6 层）
- 归一化：config 的 `data.biases/scales`（逐变量 mean/std），**换数据集必须重算**

### 模型

- `UNetDDPMVer01`：channel_mults [1,2,4,8,8]（4 次下采样，输入边长须 16 倍数），attn_res 16，res_blocks 1，dropout 0.2
- 小模型 128²/32ch(~1.2M)，大模型 192²/64ch(~4.8M)
- SI 配置：eps 0.2、quadratic、L2、n_timestep 10、`channel_weights: [1,1,10,...]`（W 分量权重 10，wloss 变体）

### 物理约束（核心创新点）

- 施加在**漂移场 b**（模型输出）上，不是最终风场上
- 散度：∇·b = 0（中心差分）；涡度：ζ_coarse(b) = 0（avg_pool2d(kernel=4) 只约束大尺度）
- config 开关：`si.divergence_weight: 0.1`、`si.vorticity_weight: 0.05`（0 = 关闭；消融实验纯 L2 无物理约束）
- 实现位置：`src/dl_model/si_follmer/si_follmer_framework.py`
- **深圳完整 NS 可行**：P/PB/T（密度）、TKE/KM/KH（湍流）、PH/PHB（层高）齐全，动量方程 6 项（时间导数/平流/气压梯度/科氏/湍流扩散/连续性）全可约束——这是香港做不到的（香港 npz 无 3D 气压温度）

### 昼夜划分

- HK 用 swdown 阈值：>50 W/m² 白天，<5 夜间（存在 npz 的 swdown 字段）
- SZ 用合成 swdown（太阳几何，elev>3°≈50W）——过滤逻辑不变
- 数据加载：`config.data.day_night_filter: all/day/night`（dataloader 按 npz 的 swdown.mean() 过滤）

---

## 5. 工作流（具体命令）

### 训练

```bash
# 单个：bsub 或登录节点
python scripts/train_schrodinger_bridge_model.py \
  --config_path configs/香港-基础/config_wind_3d_final.yml \
  --experiment_name ExperimentSchrodingerBridge3dWind --device cuda:0
# 模板：lsf/最终模型/train_final.lsf（模块加载、日志目录、GPU 声明齐全）
```

### 评估

```bash
python scripts/evaluate_ablation_day_night.py \
  --checkpoint_base_dir data/DL_result/ExperimentSchrodingerBridge3dWind \
  --results_dir results/ablation_day_night --suffix _large --device cuda:0
# 说明：脚本按 5 消融 × 2 昼夜自动找配置和 checkpoint（配置在 configs/香港-昼夜消融-*/）
# 不带 --suffix = 小模型；--suffix _large = 大模型
# 输出：每模型 *_metrics.json + 控制台对比表 + all_summaries.json
python scripts/evaluate_wind_3d.py --config_path ... --checkpoint_path ...  # 单模型评估
```

### 绘图

```bash
python scripts/plot_day_night_loss.py --log_dir logs/ --output_dir results/xxx
# 注意：日志解析正则只认 "train error: avg loss = X.XXX"；支持 abl_day_all_*.out 和 abl_large_day_all_*.out
```

### 配置/提交生成

```bash
python scripts/generate_day_night_ablation_configs.py          # 小模型 128/32/400ep
python scripts/generate_day_night_ablation_configs.py --large  # 大模型 192/64/600ep
# 输入：configs/香港-全天消融/ 的 5 个基础配置；输出：configs/香港-昼夜消融-*/ + lsf/昼夜消融/训练任务/
```

### 预处理（深圳）

```bash
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/prepare_wind_data_3d_sz.py \
  --scheme myj --limit 2        # 冒烟测试
... 全量（不加 --limit；--workers 4 多进程；--skip_stats 跳过统计）
```

### 数据探查

```bash
/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python scripts/inspect_wrfout.py \
  --data_dir /fsb/home/yutingwang/share/Data_WRFout/case01_Shenzhen/meso_202007_myj --domain d04
```

---

## 6. 历史实验与结论（写报告/汇报时用，防止重复实验）

1. **全天消融（香港）**：地形 >> 热力 > 气压 ≈ PBLH
2. **昼夜消融小模型（128²/32ch）**：所有 ΔRMSE < 1%，噪声水平——模型太小 + 数据减半，条件变量贡献不显著
3. **昼夜消融大模型（192²/64ch）**：关键发现——
   - 地形白天重要（NoTerrain +0.0236）：容量足够后地形信息发挥作用
   - 热力和 PBLH 夜间是噪声（各 -0.02）：夜间 swdown≈0、边界层稳定
   - 模型容量门槛：小模型去地形反而更好（-0.0077），大模型去掉就变差（+0.0236）
4. **昼夜差距稳健**：夜间 RMSE 比白天低 ~24%，W 分量低 33-40%，与尺度/配置无关
5. RMSE 为标准化值（越小越好）；ΔRMSE 正 = 该变量重要

---

## 7. 仓库结构（2026-08-24/25 整理后）

```
├── src/                    # 核心代码（dl_config/dl_data/dl_model/dl_utils）
├── configs/                # 按历史分类：模板/ 香港-基础/ 香港-全天消融/
│                           #   香港-昼夜消融-小模型/ 香港-昼夜消融-大模型/（深圳/待建）
├── scripts/                # 现代脚本（训练/评估/绘图/预处理/生成器）
├── lsf/                    # LSF 提交：最终模型/ 昼夜消融/（训练任务/ 为生成器输出）
├── results/                # 评估 JSON + 图
├── 组会汇报/               # 6-11 最终模型 / 7-17 昼夜消融 / 历史/
├── docs/                   # 项目历史与数据说明.md + 本文件
├── prepare_wind_data_3d.py # 香港预处理（历史，保留作参考）
├── data.list               # 深圳数据集说明（老师提供）
└── data/  logs/  prepare_npz_*   # 服务器上的大数据（gitignored，本仓库是 git 同步的）
```

---

## 8. 常见坑（血泪教训）

1. **f-string/中文在服务器 Python 2 上直接挂**——脚本要 Py2/3 兼容 + ASCII
2. **GPU 队列 4 分钟杀 CPU 任务**——数据任务 nohup 登录节点
3. **screen 会话**：命令回显后输出还没出来，多等几秒再 hardcopy；`stuff` 里用 `$'...\n'` 发换行
4. **SSH 会掉**——掉线重连即可，nohup 任务安全
5. **GitHub push 超时**——加 `http_proxy=http://127.0.0.1:7892` 代理重试
6. **配置路径**：configs/lsf 已按历史分目录，脚本里路径已同步更新；**新增脚本引用配置时用新路径**（如 `configs/香港-基础/...`、`lsf/昼夜消融/训练任务/...`）
7. **不要动 `data/`、`logs/`、`prepare_npz_*`**（gitignored 大数据，训练依赖）
8. **用户个人文件**：根目录 `简历/` 是用户私人物品，不要动
9. **normalization**：换数据集必须重算 biases/scales，否则 RMSE 没有意义
10. **评估脚本的 checkpoint 路径**：`data/DL_result/ExperimentSchrodingerBridge3dWind/<配置名>/checkpoint.pth`，配置名 = yml 文件名

---

## 9. 与用户协作方式

- 用户用中文交流；回复用中文
- 服务器密码/OTP 由用户输入，agent 不要尝试
- 用户授权 agent 直接 commit + push（需要时），push 用代理
- 大模型训练动辄数天：提交后设提醒定时查 bjobs，不要阻塞
- 组会报告写好后放 `组会汇报/<日期>/`，配图用绝对路径引用同目录 png
