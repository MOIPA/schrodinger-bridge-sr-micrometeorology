---
name: pptx-editing
description: 在已有 PPTX 模板上改版/续页的实操规范与踩坑记录——字体未嵌入导致的替换现象、版式(slideLayout)内嵌图片造成的 logo 与底部装饰重复、QuickLook 只给首页缩略图导致"逐页目检"失效、模板标题字号不统一、按模板基准页对齐的量测方法。当需要修改既有 PPTX（改字号/字体、增删页、对齐模板版式、排查视觉不一致）时使用。
---

# PPTX 改版规范与踩坑记录

> 来源：2026-10-14 中期汇报改版（`汤子晴-中期汇报-v3.pptx` → `v4.pptx`，12 页扩到 23 页）。
> 本文记录的是**实测踩过的坑**，每条都对应一次返工。改 PPTX 前先读第 4 节。

## 1. 工作方法：构建脚本 + 逐页渲染 + 像素量测

**不要**直接用 python-pptx 逐个改文件（不可复现）。固定三件套：

```bash
python3 build_ppt_v4.py      # 生成（含字体统一、版式清理、前五页规范化）
python3 render_slides.py deck.pptx 6 12   # 逐页渲染到 /tmp/ppt_render/slide_XX.png
python3 check_layout.py  deck.pptx        # 几何校验：出界/图文重叠/文字溢出
python3 check_title_fit.py                # 标题是否折行
```

**判断版式是否与模板一致，必须做像素量测**，不能靠坐标推断。例：

```python
# 量标题/logo 的墨迹包围盒（渲染图 1600px 宽 -> 120 px/in）
im = Image.open(png).convert('RGB'); a = np.array(im); px = w / 13.3333
ink = (a.astype(int).sum(axis=2) < 600)      # 非白即墨
rows = np.where(ink.any(axis=1))[0]; cols = np.where(ink.any(axis=0))[0]
print(x0/px, x1/px, y0/px, y1/px)            # 与模板页逐项比
```

## 2. 坑 1：字体未嵌入 = 每台机器替换结果不同（最耗时的一个）

**现象**：同一份 PPT，本机渲染和用户 WPS 里看到的字体不一样；标题"看起来大小不同"。

**根因**：v3 的正文用 `Quattrocento Sans`、主题主字体是 `微软雅黑`，**本机都没装、PPT 也没嵌入**（`ppt/fonts/*` 不存在、`presentation.xml` 无 `embeddedFontLst`）。渲染器只能做字体替换，**不同软件替换成不同字体**。

**排查**：
```python
import zipfile
z = zipfile.ZipFile(deck)
[n for n in z.namelist() if 'font' in n.lower()]        # 空 = 未嵌入
'embeddedFontLst' in z.read('ppt/presentation.xml').decode()   # False = 未嵌入
```
再逐 rPr 统计实际 typeface：`slide.shapes._spTree.iter('{...}a:rPr')`。

**修复**：把**全册**（含 `ppt/slides`、`slideLayouts`、`slideMasters`、`notesSlides`、**`theme`**）的
`a:latin / a:ea / a:cs` typeface 与主题 `majorFont/minorFont` 显式改写为一种跨平台字体（选 Arial）。
注意：**只改 slides 不够**，版式与主题会继续做替换。

```python
xml = re.sub(r'(<a:(?:latin|ea|cs)\b[^>]*\btypeface=")([^"]*)(")',
             lambda m: m.group(0) if m.group(2) in protect else m.group(1)+target+'\\"', xml)
```

**教训**：用户说"标题 AA 和 BB 不一样大"时，**先问用户用的什么软件**。
本例用户在 WPS 里看到前五页标题是 Arial 28pt —— 那正是 `Quattrocento Sans` 被替换成 Arial 的结果。
所以正确做法是**把标题显式设成 Arial 28pt** 去匹配用户的观感，而不是找一个"最像原字体"的替代品
（我曾选 `Heiti SC`，反而与用户看到的不一致而返工）。

### 字体选择的取舍（实测）
| 目标 | 做法 |
|---|---|
| 与用户 WPS 观感一致 | 直接用 Arial（WPS/PowerPoint 对西文模板字体的常见替换结果） |
| 写字面最像模板的替代 | 需实测：同一页换 10 种字体各渲染一次，比 ink 高度（本例 Heiti SC 0.542 vs 模板 0.567 in，微软雅黑替换后仅 0.425） |
| 避免再被统一脚本刷掉 | `unify_font.convert(..., protect=(TITLE_FONT,))`，并在其后**再写一次**标题字体字号 |

## 3. 坑 2：版式(slideLayout)内嵌图片造成"重复 logo / 底部重复"

**现象**：
- 右上角**两个**南大 logo 叠在一起
- 底部装饰"重复"（表现为**两层校园剪影**，中间一条接缝）

**根因**：`ppt/slideLayouts/slideLayout7.xml`（空白版式）里**内嵌了两张图**：

| 版式内嵌图 | 位置 | 后果 |
|---|---|---|
| `图片 4` | L10.75 T0.31 W2.29 H0.73 | 与幻灯片页眉组里的 logo（W1.64）重叠，且**更大 38%**，会盖住后者 |
| `图片 8` | L0.00 T5.10 W13.33 H1.46 | 与幻灯片自带的底部装饰条（T6.57）叠成两层 |

**修复**：构建时从版式里移除这两张内嵌图（改一次，所有用该版式的页都干净）：
```python
blank = get_layout(prs)
for sh in list(blank.shapes):
    if sh.shape_type == 13 and sh.name in ("图片 4", "图片 8"):
        sh._element.getparent().remove(sh._element)
```

**诊断手法（关键）**：把两页的**同一区域裁剪放大并排**，一眼就能看出"两层/两个"。
坐标测量容易漏，因为两张图都是浅色半透明。

```python
p5.crop(box).resize(...).save('a.png'); p6.crop(box).resize(...).save('b.png')   # 并排
```

**注意区分"该删哪一层"**：
- 版式的 `图片 4`（大 logo）是**多余**的 → 删；
- 页眉组里的小 logo（W1.64，与模板页眉一致）是**正确**的 → 留；
- 底部装饰条（`图片 119`，T6.57）是**正确**的 → 留；版式的剪影（T5.10）是**多余**的 → 删。
反了就会出现"删掉了正确的那一个，留下重复的"。

## 4. 坑 3：QuickLook 只给首页缩略图 → "逐页目检"是假阳性

**现象**：写了个"逐页轮转首页再渲染"的脚本，以为逐页看过，实际每次渲染出来**都是第 1 页**。
所以重复 logo、标题压线、图被裁这些问题全都漏过，连续三轮返工。

**根因**：`qlmanage -t deck.pptx` 直接返回 PPTX **内嵌的 `docProps/thumbnail.jpeg`**（首屏缩略图）。

**修复（`render_slides.py` 已固化）**：
1. **先剥离** `docProps/thumbnail.jpeg`（重写 zip）；
2. 把目标页 `sldIdLst` 移到首位；
3. 保存到临时文件 → 再剥离一次（python-pptx 保存会带回缩略图）→ 才 `qlmanage`。

```python
def strip_thumb(src, dst):
    zi = zipfile.ZipFile(src); zo = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
    for it in zi.infolist():
        if it.filename == 'docProps/thumbnail.jpeg': continue
        zo.writestr(it, zi.read(it.filename))
```

**验证方法**：让脚本渲染两张**内容明显不同**的页，若输出图相同 → 渲染方法坏了。

## 5. 坑 4：模板标题字号本身不统一；放大字号会折行压内容

- 模板第 3、4 页标题 **20pt**，第 5 页 **28pt**（v3 原始状态）。
- 把第 3、4 页提到 28pt 后，标题**折成两行并压住正文**（这两页标题长 89/90 字符）。
  → 回退：模板页不动字号，只统一新页。
- **判断标题会不会折行**要按实测标定，不要凭感觉设系数：
  实测 Arial 约 **0.53 em/字符**（45 字符 × 28pt → 渲染宽 9.25 in）；
  用 0.60 em 会误报一堆正常的标题。`check_title_fit.py` 已用此值。

## 6. 坑 5：其他零散

- **页脚占位符**：版式里有 `日期/页脚/幻灯片编号` 占位符，**不填充则不可见**，不必删。
- **页码**：模板页**没有**页码；新页若自加页码，用户会认为"多出来的页标"。默认不加。
- **节号小方块 "01"**：位于 L12.27 T0.29、配色 `#637064`（青灰），会压在 logo 下方，属无用装饰 → 删。
- **`add_pic` 的宽高比**：只给 `width` 时 python-pptx 会按原图比例算高度；给固定 `height` 会**拉伸**。
  改用 `fit_pic(slide, fname, l, t, max_w, max_h)`：读原图宽高比，按可用框等比缩放，绝不拉伸/越界。
- **换图必须重算尺寸**：本次老师新图宽高比有变化（`fig_p2_losses` 1.86→2.43、`fig_p3_bars` 2.00→1.69），
  沿用旧宽度会越界或拉伸。换完必须重跑 `check_layout.py`。
- **横长图（AR>3）在左侧半栏会很小**：概念图/流程图若上下留白多，宁可**重画成紧凑横排**
  （等高卡片 + 一条流程箭头），或改为**图占满整宽、说明移到图下方**的两栏。
- **底部装饰条会遮内容**：装饰条占 T6.57–7.71；正文下限取 **≤6.4 in**。
- **构建脚本顺序**：`删除旧页` 必须在 `追加新页` 之后做，否则索引错位。
- **改完必须重跑** `add_notes.py`：**重建会清空备注**。
- **改模板里已有的文字，必须按原文本框的行容量控制长度**：python-pptx 不会自动缩排，
  超长会被 PPT **静默裁剪**（第 4 页 (b) 框 W3.40" H1.17" 在 14pt 下只能容 ~5 行/约 135 字符，
  (d) 框 W2.97" H1.81" 约 8 行/约 135 字符）。改写后**必须渲染出来看**，不能只看文本替换成功。
- **替换模板文字时注意"跨 run"**：同一句可能被拆成多个 run 或段落
  （如 `(d) Low-Altitude` + `Operations` 两段）。按 run 精确匹配会漏，需按子串匹配或整段替换。

## 6.5 插图配色：必须先提取老师调色板，不要自行判断

**教训**：我一度把 `fig_p2_losses` 的琥珀色 `#D89000` 当成"不统一"改成紫色，
后来 `check_fig_palette.py` 一跑才发现**老师自己的图里就有琥珀**
（`fig_p1_ablation` 6.1%、`fig_p2_divergence` 8.7%）。**琥珀是其调色板里的次要强调色**（对照臂/阈值基准），
不是风格错误。**先量，再改**。

老师 10 张新图实测调色板（`开题报告图/`）：

| 用途 | 色值 |
|---|---|
| 灰（baseline / 主臂） | `#7A7A7A` |
| 浅灰（柱条） | `#ADADAD` |
| 紫（低→高） | `#A66FB0` / `#8A4D96` / `#6E0F6D` |
| 浅紫过渡 | `#C9A9CE` |
| **琥珀（次要强调）** | **`#D89000`** |
| 标题文字 | `#2B2B2B` |
| 轴 / 刻度文字 | `#52514E` |
| 网格线 | `#E2E2E2` |

自查命令：`python3 check_fig_palette.py`（统计各图主色，报出离线色）。

**改插图配色时不要用 HSV 换色调**：试过把琥珀 hue 35→295，
`#D89000` 变成了品红 `#C500D8`（色相环上 295° 偏红）。
稳妥做法是**按亮度在调色板内线性映射**（保留明暗层次），或直接重绘。

## 7. 本模板的版式基准（参照页 = 第 5 页）

| 元素 | 位置/尺寸 | 字号 |
|---|---|---|
| 标题 | L0.40 T0.88 W9.88 H0.56 | 28pt 粗体 #6A005F |
| 小节小标题 | L0.47 T1.74 | 18pt 粗体 |
| 正文 | L0.47 T2.15 W5.56 | 16pt |
| 表格 | L5.60 T2.34 W6.89 | 15pt |
| 页眉分隔线 | y = 0.89（由页眉组 `直接连接符 9` 提供） | — |
| 右下 logo | x 11.35–12.99，宽 **1.64 in** | 页眉组 `图片 6` |
| 底部装饰条 | T6.57–7.71，校园剪影（**不含 logo**） | — |

**新页的标题**：`L0.40 T0.34 W10.10`，28pt Arial（W10.10 是避开 logo 的上限）。
**新页内容区**：`T1.78 – 6.40`。

## 8. 交付前检查清单

- [ ] 全册字体统计只剩目标字体（含版式/主题）
- [ ] 页眉 logo 与模板页**像素量测**一致（位置 + 宽度）
- [ ] 底部只有**一层**装饰（与模板页底部区域做像素差，应 < 1）
- [ ] 无多余页码 / 节号方块
- [ ] `check_title_fit.py` 无新增 WRAP
- [ ] `check_layout.py` 除模板自带项外无告警
- [ ] `check_fig_palette.py` 全部 OK（插图为同一套配色）
- [ ] 换图后已按新宽高比重算投放尺寸
- [ ] `add_notes.py` 已重跑，备注齐全
- [ ] 逐页渲染并**真的看过图**（不是只看脚本通过）
- [ ] 原文件（v3）保持未改动

## 9. 相关脚本

| 脚本 | 作用 |
|---|---|
| `build_ppt_v4.py` | 构建：继承模板 → 清版式冗余 → 加新页 → 统一字体 → 前五页规范化 |
| `unify_font.py` | 全册字体统一（支持 `protect` 保护名单） |
| `render_slides.py` | 逐页渲染（先剥缩略图） |
| `check_layout.py` | 出界 / 图文重叠 / 文字溢出 |
| `check_title_fit.py` | 标题折行检查（0.53 em/字符，已标定） |
| `add_notes.py` | 写口播备注与配时（**重建后必须重跑**） |
| `make_ppt_figs.py` | 生成新增图（数据取自项目真实结果 JSON；配色同老师调色板） |
| `check_fig_palette.py` | 检查各插图配色是否落在统一调色板内 |

## 10. 标题与页面取舍（面向导师的汇报）

- **大标题要短**：`Data Pipeline`、`Truth Diagnostics`、`Roadmap`、`Backup`。
  把 "What / Where / Why..." 式长句标题、以及"(Phase 3 onward)"之类括注去掉——
  长句标题是明显的"AI 味"，导师会觉得啰嗦。小节点（prompt 行）才放"一句话结论"。
- **模板页的标题同样要改**：v3 原有标题又长又绕
  （`High-quality meteorological support is urgently needed for low-altitude flight safety`），
  改为 `Background` / `Cascading Scales` / `Low-Altitude Operations`。
- **不要为"没做过的阶段"各占一页**：阶段 4/5/6 尚未开始、内容单薄，摊成多页会显得简陋。
  合并成一页 `Roadmap`（含未回传的架构对比状态），比五页独立空页更有说服力。
- **单一结果不要单独成页**：如"三种子稳健性"只有一个柱状图，独立成页很突兀；
  把该结论写进对应结论页的讲稿与要点里（必要时作为该页的一条 bullet）。
- **模型输入输出要有专门一页**：中期汇报里这是核心，需写清
  条件通道数、状态通道数、网络与输出维度、以及交付/评估空间。

## 11. 内容一致性核对（写幻灯前必做）

**踩过的坑**：真值诊断页我凭印象写了一条"C-grid 散度残差设定物理约束阈值"的结论，
但该页配的是动能谱图，**图里根本没有这个结论** —— 属于把不相干内容写进页面。

**规矩**：
1. **每一页的每个断言，都要能在配图或阶段报告里找到出处**。图与文必须互相支撑：
   配图是动能谱，就只能讲有效分辨率；散度残差没有配图，就别写成图注式的结论。
2. 写页面前**先读一遍该阶段的报告**（`组会汇报/*/组会汇报-*.md` 或 `docs/阶段报告/`），
   把报告的"几项结论"直接作为该页的 bullet 骨架，不要自己重新组织。
3. 数字优先用报告里的**实测值**（如散度残差除以密度后 3.9e-4 ~ 9.9e-4 s⁻¹），
   不要用"量级合适"这类含糊说法。
4. **跨页查重**：同一数字/机制只在一页作为主证据，其它页最多引用一句并指向它。
   本项目实际查出的重复：
   - 散度约束的数值与机制曾同时出现在"损失对比页"与"散度页" → 损失页只留对比，散度页讲机制；
   - 主指标/切变四个数字曾同时出现在柱状图与右侧文字栏 → 删掉文字栏，图承担数字；
   - 档位、网格、层数曾同时出现在"模型 I/O 页"与"数据管线页" → 前者讲接口，后者讲样本与划分。

   查重脚本（对关键词逐页统计，>2 页出现即为可疑）：

   ```python
   slides = [' '.join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame).lower()
             for s in Presentation(deck).slides]
   for k in ["0.9574", "23 layer", "log z0", "divergence"]:
       print(k, [i+1 for i, t in enumerate(slides) if k in t])
   ```

## 12. 插图不要跨页复用

**踩过的坑**：`fig_supervision_space.png` 同时用在"评估空间""在哪个空间算损失""监督方式对比"三页，
其中后两页讲的是不同的东西，同一张图既重复又讲不清。

**规矩**：
1. **一图一页**。若两页需要表达不同重点，就画两张图（本项目为"两种监督方式"专门画了
   `fig_two_methods.png`：两行对比"网络输出 → 损失 → 评分空间"）。
2. 概念图里**不要出现没有标注的元素**。原图左侧那个框画了 11 条等距横线却没有任何说明，
   看不出是"模式层"（虚线形状还容易被误读为网格线）→ 必须加框内标注
   （"23 terrain-following model layers / each line = one layer"），并给文字加白色底
   （`bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.92)`）以压住线条。
3. **自查脚本**（统计每张图被哪些页使用，>1 页即需说明理由）：

   ```python
   R='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
   use=defaultdict(list)
   for i,s in enumerate(Presentation(deck).slides,1):
       for sh in s.shapes:
           if sh.shape_type!=13: continue
           for b in sh._element.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
               rid=b.get(R)
               if rid: use[str(s.part.related_part(rid).partname).split('/')[-1]].append(i)
               break
   ```

## 13. 对外材料不要出现内部编号

**踩过的坑**：Roadmap 页写成 `Phase 3 remainder / Phase 4 / Phase 5 / Phase 6`，
封面与创新点页也写了 `Phase 0 to the front part of Phase 3`、`Phases 0-2 complete`。
这些是项目组内部的阶段编号，**答辩老师在开题报告里看到会觉得莫名其妙**。

**规矩**：
1. 对外页面用**内容名**代替编号：`Finish the method comparison` / `Multi-scale downscaling` /
   `Time downscaling` / `Generalisation and anchoring`。
2. 内部说法同样要清掉：`wave 1`（改"the first training runs"）、`frozen recipe`（改"the configuration
   established so far"）、`T3.4` 之类任务号。
3. **备注也要一起清**——备注虽不投屏，但打印或转发时同样暴露内部编号。
4. 交付前扫描（幻灯片 + 备注一起查）：

   ```python
   import re
   for i, s in enumerate(Presentation(deck).slides, 1):
       for sh in s.shapes:
           if sh.has_text_frame and re.search(r'Phase\s*[0-9]|wave 1|frozen recipe|T[0-9]\.[0-9]', sh.text_frame.text):
               print('slide', i)
       if s.has_notes_slide and re.search(r'Phase\s*[0-9]|wave 1|frozen recipe|T[0-9]\.[0-9]', s.notes_slide.notes_text_frame.text):
           print('notes', i)
   ```

5. **删页时同步删备注块**：删掉"前序管线""问答备用"两页后，备注字典里残留了对应块导致重复键，
   必须一起清理（`add_notes.py` 会因键数不匹配而报错，这是好事，别绕过它）。

## 14. 幻灯是给答辩老师看的，不是给自己看的

**踩过的坑**：第 11 页写了一栏 `How to read it`，内容是
"Each point is the change in the main metric… / Green means… red means…"。
这是**操作说明**，答辩老师不需要被告知怎么读图。

**同类问题（都要清）**：
| 类别 | 反例 | 改法 |
|---|---|---|
| 读图说明 | "How to read it"、"Green means… / red means…"、图例文字 | 删掉；图自带的图例已足够。改为讲**试验设计**（多少 run、控制什么、显著性怎么判） |
| 教学口吻 | "Four diagnostics that fix what the model should and should not be asked to learn" | 改为陈述事实："Four properties of the truth fields that shape the experiment design" |
| 课堂提问 | "Why the delivery space cannot simply be the model space" | 改为结论句："Delivery levels differ from model layers, so the evaluation space must be defined explicitly" |
| 教学对比 | "instead of the whole field"、"No incremental input helped" | 直接给结果："cuts the main metric from X to Y"；小标题改中性："Incremental inputs" |
| 自我辩解 | "A formal negative result"、"Not a failed experiment: a clean statement that…" | 改为中性解释："Each constraint improves its own diagnostic while degrading point accuracy, so none enters the final loss" |
| 口语化 | "so the hinge does its job"、"fixed the rule that…" | "confirming the discretisation and the calibration"、"every result reported here comes from runs that converge cleanly" |
| 自我指涉 | "the per-level view on the next slide matters" | 改为信息性："the per-level breakdown follows on the next slide" |
| 一句话说两遍 | 图下 caption 与上方 prompt 同义 | 二者分工：prompt 给结论，caption 说明**图里画的是什么**（指标、单位、相对基准） |

**注意区分**：**演讲备注（notes）里可以有** `do not read out`、`If asked about…` 这类提示，
那是给讲者自己的；**投屏正文里不能有**。

**自查正则**：
```python
PAT = r'how to read|read it|green means|red means|we claim|not a failed|does its job|should be asked|Phase\s*[0-9]|Stage-?[0-9]'
# 只扫 sh.text_frame（幻灯片正文），不要扫 notes_slide
```

**关于图内的内部编号**：老师自制图里保留 `Phase 1/2/3` 属于原始产出，一般不必改；
但**幻灯片正文的小节提示行、标题、脚注里出现 Phase/Stage 编号必须清掉**。

## 15. 信息量大的图要单独成页，并给足高度

**这次的处理**：一张四联个例图（d04 真值 / 粗端输入 / 基线 / 联合监督，带色标与区位插图）
原本挤在"空间分布"页右下角，细节完全看不清 → 拆成两页：
- `Spatial Structure` 只讲四组误差柱状图；
- `Strong-Wind Case` 单独一页放大个例图。

**关键坑：近方形图受"高度"限制，不是受宽度限制。**
内容区高约 4.6 in，图 AR≈1.24 → 最大宽只有 4.6×1.24 ≈ **5.7 in**，
此时再写 `max_w=7.6 / 8.3` 都没用（`fit_pic` 会按高度封顶）。
**要放大就得腾高度**：把标题下的 `prompt` 行去掉（信息移入右侧文字栏），
内容区顶部从 1.78 in 提到 1.73 in 并允许图占满到 6.28 in —— 图宽随之从 5.7 增到约 6.8 in。

**通用规则**：
- 图 AR < 1.6（偏方/竖版）→ 只能靠**腾高度**放大；必要时去掉 prompt 行。
- 图 AR > 2.5（横长）→ 受宽度限制，可考虑"图占满整宽 + 说明移到图下方"。
- 分幅带 (a)(b)(c)(d) 标号的图，**每个分幅的标号必须可读**才算放大到位。

## 16. 图里的数字必须回配置/代码核对，不能抄文档

**踩过的坑**：模型 I/O 图上写"input 146 channels = 72 state + 74 condition"，
来源是 `docs/大纲阶段0_协议冻结.md` 的**冒烟测试段**——那段文档自己注明是"修复中途版本"。
实际 V* 配置（`configs/深圳/phase3/config_wind_canvas_p3_joint.yml`）是 **in_channel: 167**
= 72 state + **95** condition。

**规矩**：
1. 通道数、参数量这类数字，**从配置文件或代码反推**，不要抄文档里的历史值：
   ```bash
   grep -m1 "in_channel:" configs/深圳/phase3/config_wind_canvas_p3_joint.yml   # 167
   # 条件通道 = in_channel - out_channel
   # 再用 src/dl_data/dataset_wind_canvas.py 的 build_input_channel_names() 逐组核数
   ```
2. 逐组通道数要能**加起来等于总数**（本项目：46+23+1+1+24 = 95）。
   若加不上，说明某个组的定义理解错了（曾误以为 `fine_geom` 是 3 通道，实为 1 HGT + 23 层 = 24）。
3. 图上**每组都标出通道数**，读者才能自己验算；只写总数会被问"74 都是什么"。

## 17. 用户手工调过的尺寸要固化进脚本

用户在 PPT 里手工拖过图片大小后，**这些改动只在 PPTX 里，构建脚本不知道**；一重建就被覆盖。

处理流程：
1. 先读当前 PPTX 里各图的实际框（`sh.left/top/width/height`），找出与脚本不同的页；
2. 把用户的值写回脚本（必要时直接按框放置）；
3. 再重建。

**`fit_pic` 的局限**：它默认等比缩放，宽高比变化后会按高度封顶、图反而变窄。
需要"就按我给的这个框放"时用 `force=True`（允许轻微纵向压缩），例如：
```python
fit_pic(sl, "fig_model_io.png", 0.50, 1.67, 12.05, 3.22, force=True)
```

## 18. 通道数要能"逐项加得起来"，且注意交错网格的层数差

**这次被追问的点**：图上写"72 state channels"，但没说 72 从哪来。

正确拆法（本项目，Arakawa C 网格原生交错）：

| 分量 | 网格位置 | 通道 |
|---|---|---|
| U | 质量层，逐 23 层 | 23 |
| V | 质量层，逐 23 层 | 23 |
| **W** | **界面层，逐 24 层** | **24** |
| U10、V10 | 近地面诊断量 | 2 |
| 合计 | | **72** |

**易错点**：**W 比 U/V 多一层**。U/V 在 half level（质量层），W 在 full level（界面层），
界面数 = 质量层数 + 1（本项目 23 层 → 24 界面）。写"23 层"时若不区分，
U/V/W 的通道数就加不上（23+23+23+2 = 71 ≠ 72）。

**规矩**：凡是给出通道总数，旁边就要能**逐项列出并加总**；
涉及交错网格时分别写清"质量层 / 界面层"。条件通道同理：
`46 + 23 + 1 + 1 + 24 = 95`，`95 + 72 = 167`。

