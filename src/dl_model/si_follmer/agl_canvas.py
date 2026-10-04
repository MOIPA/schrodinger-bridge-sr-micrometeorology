# -*- coding: utf-8 -*-
"""阶段 3:可微 AGL 高度插值层(纯函数、batch-first、无副作用、无可学习参数)。

数学与 scripts/outline/60_agl_operator.py 的 numpy `agl_interp`(L35-53,评估侧
唯一实现)逐行一致:对每个目标 AGL 高度 t 用插值表 (idx, w)(w 为 ln z 线性权重),
    idx >= 0 : val = w·f[k] + (1−w)·f[k+1]      (模式层 k 与 k+1 之间夹逼)
    idx == −1: val = w·f[0] + (1−w)·f_10m       (低于首层,10 m 通道锚定)
    idx == −2: val = f_10m                      (目标即 10 m,直接取 10 m 通道)
field10=None 时不做 −1/−2 分支,idx 一律 clamp 到 [0, nlev−2] 后线性插值。
插值表由 1 km 静态 z_agl 逐像素构造(build_agl_table),预测与真值共用同一张表,
表的值域(-2..nlev-2)与网格对齐由"dataset 用同一次 crop"的契约保证,本层
不做任何 offset/clamp 之外的改写。

张量口径:
    field   (B, nlev, ny, nx)
    idx     (B, nt,   ny, nx)  int64,取值 −2..nlev−2
    w       (B, nt,   ny, nx)  与 idx 同形
    field10 (B, ny, nx) 或 None
    返回     (B, nt,   ny, nx)

所有函数均为 torch 张量运算、可微(field/field10 侧有梯度)、无副作用。
"""
import torch

__all__ = ["agl_interp_batched"]


def agl_interp_batched(field, idx, w, field10=None):
    """field (B,nlev,ny,nx);idx (B,nt,ny,nx) int64,取值 -2..nlev-2;w (B,nt,ny,nx);
    field10 (B,ny,nx) 或 None。返回 (B,nt,ny,nx)。

    逐层(nt 循环):kc=idx[:,k].clamp(0,nlev-2);lo=gather(field,1,kc.unsqueeze(1));
    hi=gather(field,1,(kc+1).unsqueeze(1));val=w[:,k]*lo+(1-w[:,k])*hi;
    若 field10 非 None:idx==-1 处 val=w[:,k]*field[:,0]+(1-w[:,k])*field10;
    idx==-2 处 val=field10。stack 成 (B,nt,ny,nx)。
    """
    nlev = int(field.shape[1])
    nt = int(idx.shape[1])
    idx = idx.to(device=field.device).long()
    w = w.to(device=field.device, dtype=field.dtype)
    if field10 is not None:
        field10 = field10.to(device=field.device, dtype=field.dtype)

    outs = []
    for k in range(nt):
        kc = idx[:, k].clamp(0, nlev - 2)
        lo = field.gather(1, kc.unsqueeze(1)).squeeze(1)
        hi = field.gather(1, (kc + 1).unsqueeze(1)).squeeze(1)
        wk = w[:, k]
        val = wk * lo + (1.0 - wk) * hi
        if field10 is not None:
            # 与 numpy 版同序:先处理 −1(10 m 锚定的最低层延伸),再处理 −2(目标即 10 m)
            val = torch.where(
                idx[:, k] == -1, wk * field[:, 0] + (1.0 - wk) * field10, val)
            val = torch.where(idx[:, k] == -2, field10, val)
        outs.append(val)
    return torch.stack(outs, dim=1)
