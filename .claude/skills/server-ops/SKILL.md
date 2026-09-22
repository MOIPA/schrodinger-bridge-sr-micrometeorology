---
name: server-ops
description: LSF 集群服务器操作手册——screen 共享会话工作流、conda 环境选择、GPU 队列规则（PEND 必须换队列、避免卡死队列）、bsub 提交模板与内部日志技巧、setsid 长任务、GitHub 同步（SSH push）、CPU/GPU 任务分工。当需要登录服务器跑任务、提交训练/评估、排查任务 PEND/失败、或在服务器与本地间同步时使用。
---

# 服务器操作手册(LSF 集群)

## 1. 连接方式:screen 共享会话

- SSH 是**双层密码**(第一层固定、第二层动态 OTP),agent 无法自己 ssh;由用户输入
- 工作流:用户在本地开 screen → 里面 ssh → `Ctrl-A D` 脱离;agent 用
  - 发命令:`screen -S <会话名> -X stuff $'命令\n'`
  - 读输出:`screen -S <会话名> -X hardcopy /tmp/x.txt; tail -20 /tmp/x.txt`
- 会话名带 PID 前缀(如 `5893.srv`),用 `screen -ls` 查;SSH 常每 ~40 分钟断一次,断了让用户重连
- 保活:周期性向会话发命令(如 5 分钟一次)可延长寿命,但**不保证不断**——重要任务不要依赖
- 安全层提示:自动模式可能拦截不常见或复合的 screen 命令;保持每条命令**单一目的、简单**容易通过

## 2. 环境

| 用途 | env | 说明 |
|------|-----|------|
| 训练/评估 | `wind3d` | 有 torch/cuda;**无 netCDF4** |
| 数据探查/预处理 | `pytorch-gpu`(`/fs00/software/anaconda/3/envs/pytorch-gpu/bin/python`) | **有 netCDF4**,py3 |
| 登录节点默认 | python 2.7.5 | 无 netCDF4 |

**脚本规则:Py2/3 兼容(无 f-string、纯 ASCII)**——除非确认只跑 py3 env。
激活:`module load anaconda/3 && source activate wind3d`

## 3. LSF 提交

```bash
bsub -q <队列> -gpu "num=1:mode=exclusive_process" -n 4 -R "rusage[mem=32000]" \
  -J 任务名 -o logs/任务名_%J.out \
  "cd ~/schrodinger-bridge-sr-micrometeorology && module load anaconda/3 && module load cuda/11.8.0 && source activate wind3d && python -u 脚本.py > logs/内部.log 2>&1"
```

**三条铁律**:
1. **GPU 队列会杀不用 GPU 的任务(约 4 分钟 SIGTERM)**——纯 CPU 数据任务不上 bsub
2. **PEND 就立刻换队列,不干等**(用户明确要求)。探测:
   `for q in 83a100ib 62v100ib 72rtxib e5v4p100ib 6148v100ib 7552v100 7k83; do echo -n "$q "; bqueues -w $q | tail -1 | awk '{print $9,$10}'; done`
   (第 9、10 列 = PEND、RUN;PEND=0 RUN=0 最理想)
3. **必须用内部日志重定向**(`python -u ... > logs/x.log 2>&1`)——LSF 的 -o 文件可能只保存 job summary,**python 的 stdout/stderr 会丢失**,排查失败时全靠内部日志

队列经验(2026-09 实测):83a100ib 常年 PEND 100+;72rtxib 会从空闲突变为满;**7552v100 曾整体卡死(12 PEND 0 RUN 的主机不可用状态)**;6148v100ib、62v100ib 多次成功;9654p6000ib 的 P6000 卡不兼容,避开。

## 4. 长任务与 CPU/GPU 分工

- 登录节点长任务:`setsid nohup 命令 > logs/x.log 2>&1 < /dev/null &`——**普通 nohup 在 SSH 断线时会被杀**(进程组清理),必须 setsid
- **推理一律走 GPU**:19M 参数扩散模型评估 1384 样本,GPU ~1-2 分钟,CPU **1 小时以上**——不要用 CPU 跑模型推理,只用来跑纯数据脚本(预处理、零模型统计等)
- 长任务进度:写日志到文件,agent 定期 tail;不要依赖屏幕输出

## 5. GitHub 同步

- 远程:`github.com:MOIPA/schrodinger-bridge-sr-micrometeorology`(main)
- 本地 https push 失败(网络)时改用 SSH URL:
  `git push git@github.com:MOIPA/schrodinger-bridge-sr-micrometeorology.git HEAD:main`
  (本地代理 `http://127.0.0.1:7892` 时有时无,SSH 通道通常可用)
- 服务器端 pull 用 `git pull --no-rebase`(双方都有提交时);服务器推送走 SSH 正常
- 服务器有未推送提交时,本地 pull 后再 push

## 6. ops/queue 任务脚本模式(项目惯例)

- 新任务写 `ops/queue/NN_名字.sh`:执行命令 → 结果写 `ops/result/NN_*.txt` → 末尾自动
  `cd ops && git add . && git commit -m "result NN" && git push`
- **注意:自动回传只覆盖 ops/ 目录**——`results/` 下的评估 json 不会被提交,需要用户或 agent 手动
  `git add results/... && git commit && git push`
- 手机/用户执行:`bash ops/queue/NN_名字.sh`

## 7. 数据与磁盘

- 数据在 `/fsb/home/yutingwang/share/`(共享,老师提供);项目在 `~/schrodinger-bridge-sr-micrometeorology/`
- `/fsb` 空间极大(百 TB 级),不必担心容量
- `data/`、`logs/`、`prepare_npz_*` 已在 .gitignore;PDF 不上传

## 8. 关联

- 数据清单见 sz-data-inventory skill;代码结构见 code-map skill
- 历史备忘:`docs/项目历史与数据说明.md` 的环境运维节
