# 外传 · 三尺度 3× 放大对比

> 本目录**不属于导师大纲的任何阶段**,是老师 2026-10-31 口头追加的一条支线。
> 报告、结果、代码独立存放,不写进 `docs/阶段报告/`,也不占阶段编号。
> 主项目的总入口仍是 `docs/大纲工作总览.md`。

## 一句话

问:**同样"放大 3 倍",在 27 km→9 km、9 km→3 km、3 km→1 km 三个起始尺度上,学到的算子是不是同一回事?**
老师说"每个阶段的灰度区间不一样",这句话要么被证实(三个算子必须分开训),要么被量化修正(差多少、差在哪)。

## 为什么这三段可比

WRF 嵌套是 3 倍降尺度:d01 27 km → d02 9 km → d03 3 km → d04 1 km。
三段的**放大倍数完全相同(3×)**,只有起始分辨率不同 —— 固定 ratio、只变 base scale 的干净对照。

(与阶段 4 的 27→1(9×)、3→1(3×)不重复:那两个回答"极限跨度能不能一步跨过去";
本支线回答"算子随尺度怎么变"。二者互补。)

## 实验设计

| 组 | 输入 → 输出 | 角色 | tag |
|---|---|---|---|
| A | d01 (27 km) → d02 (9 km) | 最平滑、细节最少 | `sq27to9_reg` |
| B | d02 (9 km) → d03 (3 km) | 中间尺度 | `sq9to3_reg` |
| C | d03 (3 km) → d04 (1 km) | 与阶段 3 同域,但输入换成 3 km | `sq3to1_reg` |

三组用**同一架构、同一超参**(阶段 3 的 reg 基座,只改数据路径),保证"架构差异"不是混杂因素。

## 现在的状态(2026-11-01 22:00)

| 事项 | 状态 | 证据/产物 |
|---|---|---|
| 三档细端 npz(d02/d03 作细端) | ✅ | `prepare_npz_outline_fine_d02/d03`(各 744 时次) |
| 三组静态场 + 重网格权重 | ✅ | `prepare_npz_outline_static_sq{27to9,9to3,3to1}`;画布 [121,151]/[121,151]/[100,121] |
| 划分统一到阶段 3 口径 | ✅ | `ops/debug/check_splits.py` 复核四套 split 完全一致(test 07-16T00–07-31T23) |
| 第一轮三档训练 | ⚠️ 作废 | 划分不一致,数字不可比(见 `docs/外传-3_三尺度对比_定稿.md` 顶部横幅) |
| 三档重训(统一划分) | 🔄 进行中 | 14833756 `sq27to9` / 14833740 `sq9to3` / 14833684 `sq3to1` |
| 真粗端缺口量化 | ✅ | `docs/外传-1b_真粗端的缺口.md`、`results/gap_real.json` |
| 逐层偏差追查(9→3 偏强 +0.316) | ✅ 部分 | `外传-1b §5.1`、`results/bias_*.json`(排除垂直错位,成因未定位) |
| 交叉迁移(零样本) | ⏳ 等重训 | `configs/phase3_arch/config_wind_canvas_xfer*_reg.yml` |
| 级联 27→9→3→1 | ⏳ 等重训 | `scripts/synth_coarse.py` + `queue/cascade.sh`,已 CPU 冒烟通过 |
| 一步 27→1 直连(阶段 4) | ✅ | `docs/阶段报告/阶段4a_一步27to1_首评.md`(1.0909,+28.1%) |

## 文档索引(每篇一句话)

| 文档 | 一句话 |
|---|---|
| `docs/外传-1_灰度区间量化.md` | **口径已作废**(3×3 块平均代理);留作发现问题与方法的记录 |
| `docs/外传-1b_真粗端的缺口.md` | 用真粗端 + 项目重网格权重重算缺口:R² 0.974/0.796/0.842,缺口方差占比 2.6/20.4/15.8% |
| `docs/外传-2_实验设计_三模型与插值对照.md` | 设计稿:三档同架构、与插值基线对照、交叉迁移与级联两种更硬的对照 |
| `docs/外传-3_三尺度对比_定稿.md` | **数字作废**(三档划分不同);保留"如何发现问题"的过程 |
| `docs/外传-3b_三尺度对比_统一划分.md` | 重做版;**预登记判读规则已写死**,数字等重训完成后由脚本回填 |
| `docs/外传-4_级联与迁移_草稿.md` | 级联与迁移:两端边界(上界 0.4390 / 直连 1.0909)、判定规则、逐级诊断口径 |

## 关键结论(只列已经站得住的)

1. **真粗端缺口随基尺度变粗而变大**:R² 0.974(27→9)→ 0.796(9→3)→ 0.842(3→1);
   阶段 4 的 27→1 是 0.443 —— 9 倍跨度的缺口是 3 倍跨度的 3–6 倍。配准已用"循环平移必使相关性下降"验证。
2. **9→3 的真粗端在子域内系统性偏弱 ~5%**(平均 bias +0.386 m/s,峰在急流层);不是 3× 的普遍规律
   (27→9 只有 +0.038,3→1 是 −0.053),成因指向域覆盖/地形代表性,尚未定位。
3. **阶段 4 的一步 27→1 可做**:1.0909 m/s,相对插值 +28.1%,绝对误差比 9→1 差 31.8%;
   粗端代价**随高度单调变大**(10 m +28.3% → 700 m +40.8%)。
4. **两个"更硬的对照"的判据已在跑之前写死**(级联两端边界、三尺度预登记),结果出来只许对照。

## 目录

```
sidequest_scale3x/
  README.md          本文(状态与索引)
  scripts/           数据抽取、诊断、出表
    extract_fine.py      任意域细端抽取
    gap_stats.py         3×3 块平均代理(已被 1b 取代,留作历史)
    gap_real.py          真粗端 + 重网格权重的缺口(现行口径)
    bias_by_level.py     逐层拆"粗−细"偏差(垂直错位 vs 域效应)
    scale3_tables.py     三尺度表 + 图 + json 的唯一生成入口
    casc_stages.py       级联逐级诊断 + 两端边界判定
    synth_coarse.py      把上一级预测写进下一级粗端(级联核心)
    gen_configs.py       从阶段3 reg 基座派生三档配置
    gen_cascade_cfg.py   只替换 coarse_dir 生成级联配置
    summarize3.py        单目录 summary/perhour → 表
    cmp_casc.py          多目录配对 bootstrap(级联 vs 直连)
  configs/phase3_arch/ 三档 + 两个迁移配置
  queue/               prep_data / prep_stats / train3 / eval3 / eval_xfer / cascade /
                       chain_after_train / watch / status / report4
  docs/                报告(外传-1、1b、2、3、3b、4)
  results/             产物(含 _preunify/ 存放作废的第一轮)
```

## 复现顺序

```bash
# 数据(CPU 队列即可)
RUN_LSF=1 bash sidequest_scale3x/queue/prep_data.sh
FORCE_SPLIT_FROM=prepare_npz_outline_static/split.json bash sidequest_scale3x/queue/prep_stats.sh

# 训练(训练完自动接评估/迁移/级联)
bash sidequest_scale3x/queue/train3.sh
DEPS="<三个训练作业号>" bash sidequest_scale3x/queue/chain_after_train.sh

# 出表与图(纯 numpy)
python sidequest_scale3x/scripts/scale3_tables.py ...
python sidequest_scale3x/scripts/casc_stages.py ...
```

## 自建项标注(非老师提供)

粗端 log z0(WPS 静态地理数据 + Noah Z0BRD 公式重建)、逐层 z_agl(PHB/g − HGT)、
cos(SZA)(自算天文量)、三档粗细端 npz 与重网格权重、标准化统计 —— 全部由本项目生成。详见各报告附录。
