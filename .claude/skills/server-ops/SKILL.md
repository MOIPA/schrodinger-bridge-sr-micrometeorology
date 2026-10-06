---
name: server-ops
description: LSF 集群服务器操作手册——screen 共享会话工作流、conda 环境选择、GPU 队列规则（PEND 必须换队列、避免卡死队列）、bsub 提交模板与内部日志技巧、setsid 长任务、GitHub 同步（SSH push）、CPU/GPU 任务分工。当需要登录服务器跑任务、提交训练/评估、排查任务 PEND/失败、或在服务器与本地间同步时使用。
---

# 服务器操作手册(LSF 集群)

## 1. 连接方式:screen 共享会话

- SSH 是**双层密码**(第一层固定、第二层动态 OTP),agent 无法自己 ssh;由用户输入
- 工作流:用户在本地开 screen → 里面 ssh → `Ctrl-A D` 脱离;agent 用
  - 发命令:`screen -S <会话名> -X stuff "命令"$'\r'` —— **换行必须用 `$'\r'`(CR)**,screen 不解释
    `\n`(会把字面 `\n` 打进 shell,2026-09-30 实测);也不要在 stuff 里用反斜杠转义引号,会原样送进 shell
  - 读输出:`screen -S <会话名> -X hardcopy /tmp/x.txt; tail -20 /tmp/x.txt`
  - 命令**别太长**(带 `$(...)`/多层引号的长命令会被截断或错位)——复杂逻辑写进 `ops/queue/*.sh` 再调用
- 会话名带 PID 前缀(如 `5893.srv`),用 `screen -ls` 查;SSH 会被网关按空闲掐断(表现为
  `Connection reset by peer`),断了让用户重连(把 `ssh ytw_tangzq@entry.nju.edu.cn` 打进 screen,用户只输密码+OTP)
- **保活(2026-09-30 起)**:① 本地 `~/.ssh/config` 加 `Host entry.nju.edu.cn` + `ServerAliveInterval 60`
  + `ServerAliveCountMax 6`(客户端心跳,新建连接生效);② 当前连接可在服务器侧起
  `(while true; do sleep 55; printf "\0"; done) & disown`(不可见 NUL 心跳)。仍不保证不断,重要任务别依赖
- **保活的现实(2026-10-03/04 实测)**:NUL 心跳挡不住网关,约 2-3 小时无"真实流量"仍被 reset;
  可靠做法 = **本地侧每隔 25-30 分钟经 screen 发一条真实命令**(顺带查作业状态,写成后台小脚本循环);
  断了之后把 `ssh ytw_tangzq@entry.nju.edu.cn` 预先打进 screen(用户只需输密码+OTP)。
  注意:断线后 stuff 会打进本地 shell(报 zsh 错误),从 hardcopy 里能看出来
- **stuff 必须带"提示符守卫"(2026-10-06 教训)**:用户登录过程中(Password:/2nd Password:)若 keepalive/巡检
  的 stuff 撞进去,会把命令文本打进密码/OTP 输入导致认证失败。改法:每次 stuff 前先 hardcopy 并检查末行
  是否含 `ytw_tangzq@login1`(服务器提示符),不是就不发;agent 手动巡检同理——**先只读 hardcopy 确认状态,再决定是否 stuff**
- **守卫的缺口与修复(2026-10-07)**:长任务前台运行时屏幕无提示符 → 守卫不发心跳 → 25 分钟无流量仍被网关
  reset(SIGHUP 杀掉前台脚本)。修复:登录节点长任务一律 **`setsid nohup … > logs/x.log 2>&1 &`** 与 ssh 解耦,
  agent 轮询产物文件;短命令不需要
- **断线自动重连(2026-10-07 实测)**:连接被网关关闭后,屏幕可能停在"按 ENTER 重连"提示——stuff 一个回车,
  会出现登录节点菜单(1=login1 CentOS7 LSF / 2=login2 / 3=login9),stuff `1` + 回车即可回到 login1,
  **无需重新输密码/OTP**(网关侧会话未彻底登出时)。若该路径失效才回退到"预打 ssh 命令让用户输密码"
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

**两个易踩的坑(2026-09-30)**:
- **bsub 不继承父 shell 的环境变量**:`PHASE=x bash 脚本.sh` 提交后,内层 `bsub "... bash 脚本.sh --inner"`
  拿到的是默认值 → 必须把变量写进 payload(`bsub ... "cd $ROOT && PHASE=${PHASE:-phase1} bash ... --inner"`)
- **LSF 作业依赖只能用"仍在系统里"的作业名**:`bsub -w "ended(p1_xxx)"` 对已结束(记录已清)的作业会报
  `No matching job found. Job not submitted` → 提交"训练完自动评估"这类链式作业时,只把当前 `bjobs`
  里还存在的作业名写进依赖(见 `ops/queue/59d_eval_phase1.sh --after-training`)

队列经验(2026-09 实测):83a100ib 常年 PEND 100+;72rtxib 会从空闲突变为满;**7552v100 曾整体卡死(12 PEND 0 RUN 的主机不可用状态)**;6148v100ib、62v100ib 多次成功。

**GPU 队列白名单(2026-09-30 探测)**:`e5v4p100ib`(P100)、`6148v100ib`/`7552v100`/`62v100ib`(V100)、
`83a100ib`(A100);**抢占坑(2026-10-07)**:`6148v100ib` 配置含 `TERMINATE_WHEN=PREEMPT`——高优先级作业一来
就把我们的作业杀成 `TERM_EXTERNAL_SIGNAL: job killed by a signal external to LSF`(可在启动 2 秒内被杀,
日志无 traceback、.out 里有该字样)。对策:看到该字样=被抢占而非代码问题,直接重投(或换队列;
83a100ib 当夜表现稳定);断点续训逻辑照常工作。
`9654p6000ib` 是 **RTX PRO 6000 Blackwell(sm_120)**,wind3d 的 torch 2.6+cu118
只编到 sm_90 → 提交后秒崩 `CUDA error: no kernel image is available`;`72rtxib`/`7k83` 未验证。
探测脚本:`ops/queue/59x_gpu_probe.sh` + `scripts/outline/93_gpu_probe.py`(逐队列跑一个 1 分钟 GPU 小作业)。
**挑队列规则**:只在 `bqueues` 的 PEND=0 里选 **RUN 最少**的;`6148v100ib` 曾积压 160 个排队作业把我们的
作业卡住——所以不能只看"PEND=0",还要看 RUN 数

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
- **别把 pull 静默掉**:`git pull ... >/dev/null 2>&1` 失败时看不出来,会让后续步骤跑在旧代码上
  (2026-09-30 踩过:波次闸门因 pull 静默失败而没生效)。至少 `git --no-pager log --oneline -1` 复核
- 回传结果文件后,本地用 `git ls-remote <url> main` 与 `git rev-parse HEAD` 对比确认真的推上去了
  (服务器的收尾 `git commit/push` 有失败过,需手动补一次)
- **GPU 计算节点的 PATH 可能没有 git**(2026-10-02 实测:GPU 作业末尾 `git add/commit/push` 全报
  `git: command not found`,结果回传静默失败;检查作业要顺手看 `logs/<job>_*.err`)。修法:提交时
  `GITDIR=$(dirname "$(command -v git)")`,payload 里 `cd $ROOT && export PATH=$GITDIR:\$PATH && ...`
  (60a/60d/60e/61 已带;登录节点跑的脚本不受影响)

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
