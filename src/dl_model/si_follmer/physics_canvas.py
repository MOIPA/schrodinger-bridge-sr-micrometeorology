# -*- coding: utf-8 -*-
"""阶段 2:canvas 原生 C 网格物理损失算子(纯函数、batch-first、无副作用)。

画布 y (B, 3L+3, H, W),L = n_levels = 23,通道序:
    U    通道 [0, L)             x 方向面风:画布列 j 即质量列 j 的西面
    V    通道 [L, 2L)            y 方向面风:画布行 i 即质量行 i 的南面
    W    通道 [2L, 3L+1)         L+1 个界面(垂直速度),与质量列/行同格
    U10  通道 [3L+1)             10 m 纬向风(质量点代理)
    V10  通道 [3L+2)             10 m 经向风(质量点代理)
画布比原生网格多一虚行/列(边缘复制,见 wind_canvas_statics.CanvasStatics.place);
窗口内直接用"丢最后一行/列"即取到原生交错位置,无需其它对齐假设。

物理量一律为物理单位(反标准化后)。散度口径与
scripts/outline/70_truth_diagnostics.py 的 divergence_residual 数学一致:
可压缩散度 ∇·(ρu),ρ 由质量点平均到 U/V/W 面,水平差分用网格距 dx
(真值侧 x/y 同用 dx,方形网格),垂直用逐层厚度 dz。

所有函数均为 torch 张量运算、无副作用、无可学习参数。
"""
import numpy as np
import torch

__all__ = [
    "CANVAS_STATE_LAYOUT",
    "canvas_channel_slices",
    "infer_n_levels",
    "split_canvas_state",
    "estimate_residual",
    "destagger_canvas",
    "divergence_rho_u",
    "vorticity",
    "radial_log_spectra",
    "windspeed_levels",
]


def canvas_channel_slices(n_levels):
    """按 n_levels 返回画布各分量在通道维的切片(索引规则见 CANVAS_STATE_LAYOUT)。"""
    L = int(n_levels)
    return {
        "U": slice(0, L),
        "V": slice(L, 2 * L),
        "W": slice(2 * L, 3 * L + 1),
        "U10": slice(3 * L + 1, 3 * L + 2),
        "V10": slice(3 * L + 2, 3 * L + 3),
    }


# 画布通道布局(文档,参数化于 n_levels = L);程序化切片用 canvas_channel_slices(L)。
# U/V 是面风:画布列 j(行 i)与质量列 j(行 i)共用索引,但物理位置在其西面(南面);
# W 的 L+1 个界面与质量列/行同格;U10/V10 为质量点代理(整层无垂直变化)。
CANVAS_STATE_LAYOUT = {
    "U": "y[:, 0:L]          画布列 j = 质量列 j 的西面(x 面风)",
    "V": "y[:, L:2L]         画布行 i = 质量行 i 的南面(y 面风)",
    "W": "y[:, 2L:3L+1]      L+1 个界面,与质量列/行同格",
    "U10": "y[:, 3L+1]        10 m 纬向风(质量点代理)",
    "V10": "y[:, 3L+2]        10 m 经向风(质量点代理)",
    "total_channels": "3L+3;切片见 canvas_channel_slices(L)",
}


def infer_n_levels(n_channel):
    """画布通道数 3L+3 -> 层数 L;不整除时抛错。"""
    n_channel = int(n_channel)
    if n_channel < 6 or (n_channel - 3) % 3 != 0:
        raise ValueError(f"画布通道数 {n_channel} 不是 3L+3 形式,无法推断层数")
    return (n_channel - 3) // 3


def split_canvas_state(y, n_levels):
    """画布 y (B, 3L+3, H, W) -> (u, v, w, u10, v10)。

    u, v: (B, L, H, W) 面风(保持画布形状,含虚行/列);
    w: (B, L+1, H, W) 界面垂直速度;
    u10, v10: (B, H, W) 10 m 风(质量点代理)。
    """
    L = int(n_levels)
    if y.dim() != 4:
        raise ValueError(f"画布张量应为 4 维 (B,C,H,W),实际 {tuple(y.shape)}")
    if y.shape[1] != 3 * L + 3:
        raise ValueError(
            f"split_canvas_state: 通道数 {y.shape[1]} != 3*{L}+3"
            f"(n_levels={L} 与输入不匹配)")
    sl = canvas_channel_slices(L)
    u = y[:, sl["U"]]
    v = y[:, sl["V"]]
    w = y[:, sl["W"]]
    u10 = y[:, sl["U10"]].squeeze(1)
    v10 = y[:, sl["V10"]].squeeze(1)
    return u, v, w, u10, v10


def estimate_residual(b_hat, dot_beta_t, min_t=0.5):
    """单步可微估计最终输出场 ŷ1 的残差部分,并给出可用掩码。

    quadratic 参数化下 b_true = 2t·r − ε√t·noise(r = y1 − y0),故无噪声估计
    r̂ = b̂ / dot_beta(t) = b̂ / (2t);t 越小噪声占比越大,用 min_t 排除小 t 档。

    b_hat: (B,C,H,W) 网络输出;dot_beta_t: (B,1,1,1)(或 (B,)) 的逐样本系数。
    返回 (r_hat, mask):r_hat 与 b_hat 同形;mask (B,) bool = dot_beta_t ≥ 2·min_t
    (quadratic 下即 t ≥ min_t;t=0 档因 dot_beta=0 恒被排除)。
    """
    db = dot_beta_t
    if db.dim() == 1:
        db = db.view(-1, 1, 1, 1)
    if db.shape[0] != b_hat.shape[0] or tuple(db.shape[1:]) != (1, 1, 1):
        raise ValueError(
            f"estimate_residual: dot_beta_t 形状 {tuple(dot_beta_t.shape)} 无法广播到 "
            f"b_hat {tuple(b_hat.shape[:1])} 维度的 (B,1,1,1)")
    r_hat = b_hat / db.clamp(min=1e-3)
    mask = (db >= 2.0 * float(min_t)).reshape(db.shape[0])
    return r_hat, mask


def _destagger_uv(u, v):
    """U/V 面风 -> 公共质量网格 (B,L,H-1,W-1)(内部辅助,不对外)。"""
    um = 0.5 * (u[..., :-1] + u[..., 1:])              # (B,L,H,W-1)
    vm = 0.5 * (v[..., :-1, :] + v[..., 1:, :])        # (B,L,H-1,W)
    return um[..., :-1, :], vm[..., :, :-1]


def destagger_canvas(u, v, w):
    """画布面风 -> 质量点:u (B,L,H,W)、v (B,L,H,W)、w (B,L+1,H,W)。

    u_m = 0.5(u[:, :, :, :-1] + u[:, :, :, 1:]) -> (B,L,H,W-1);
    v_m = 0.5(v[:, :, :-1, :] + v[:, :, 1:, :]) -> (B,L,H-1,W)。
    u_m、v_m 再各自裁到公共 (H-1,W-1) 以便逐点比较;w 与质量点同格(无需平均),
    同样丢掉最后一行/列的虚值,返回 (B,L+1,H-1,W-1)。
    返回 (u_m, v_m, w_m),三者均在公共质量网格 (H-1,W-1) 上。
    """
    um, vm = _destagger_uv(u, v)
    return um, vm, w[..., :-1, :-1]


def divergence_rho_u(u, v, w, rho, dx, dz):
    """可压缩散度 ∇·(ρu) 的 C 网格差分(物理单位 kg m⁻³ s⁻¹)。

    u/v: (B,L,H,W) 画布面风;w: (B,L+1,H,W) 界面垂直速度;
    rho: (B,L,H,W) 质量点密度 kg/m³;dx: float 水平网格距 m;
    dz: (L,) 或 (1,L,1,1) 的逐层厚度 m。

    面密度取相邻质量点平均、边缘单侧(与真值侧一致):
      ρ_u(i,j') = 0.5(ρ(i,j'-1)+ρ(i,j')),j'=0 用 ρ(i,0);
      ρ_v 行方向同理;ρ_w(k') = 0.5(ρ(k'-1)+ρ(k')),k'=0 用 ρ(0)、k'=L 用 ρ(L-1);
    逐质量点差分为 [F_u(j+1)-F_u(j)]/dx + [F_v(i+1)-F_v(i)]/dx + [F_w(k+1)-F_w(k)]/dz_k,
    有效 i∈0..H-2、j∈0..W-2(最后一行/列的东/北面属虚行/列),返回 (B,L,H-1,W-1)。
    dz 下限裁剪 1e-3,与 70_truth_diagnostics 的 np.maximum(dz,1e-3) 一致。
    """
    if u.dim() != 4:
        raise ValueError(f"divergence_rho_u: u 应为 (B,L,H,W),实际 {tuple(u.shape)}")
    B, L, H, W = u.shape
    for name, t, exp in (("v", v, (B, L, H, W)),
                         ("w", w, (B, L + 1, H, W)),
                         ("rho", rho, (B, L, H, W))):
        if tuple(t.shape) != exp:
            raise ValueError(
                f"divergence_rho_u: {name} 形状 {tuple(t.shape)} != 期望 {exp}")

    dz_t = torch.as_tensor(dz, device=u.device, dtype=u.dtype)
    if dz_t.numel() != L:
        raise ValueError(f"divergence_rho_u: dz 元素数 {dz_t.numel()} != n_levels {L}")
    dz_t = dz_t.reshape(1, L, 1, 1)

    # 面密度(边缘单侧)
    rho_u = torch.cat([rho[..., :1], 0.5 * (rho[..., :-1] + rho[..., 1:])], dim=-1)
    rho_v = torch.cat([rho[..., :1, :], 0.5 * (rho[..., :-1, :] + rho[..., 1:, :])],
                      dim=-2)
    rho_w = torch.cat([rho[:, :1], 0.5 * (rho[:, :-1] + rho[:, 1:]), rho[:, -1:]],
                      dim=1)                                   # (B,L+1,H,W)
    fu = rho_u * u                                             # (B,L,H,W)
    fv = rho_v * v
    fw = rho_w * w                                             # (B,L+1,H,W)

    div = ((fu[..., 1:] - fu[..., :-1]) / float(dx))[..., :-1, :] \
        + ((fv[..., 1:, :] - fv[..., :-1, :]) / float(dx))[..., :, :-1] \
        + ((fw[:, 1:] - fw[:, :-1]) / dz_t.clamp(min=1e-3))[..., :-1, :-1]
    return div


def vorticity(u, v, dx, dy):
    """相对涡度 ζ = ∂v/∂x − ∂u/∂y(四点交错差),质量点 (B,L,H-1,W-1)。

    ∂v/∂x = 0.5[(v(i,j+1)+v(i+1,j+1)) − (v(i,j)+v(i+1,j))] / dx;
    ∂u/∂y = 0.5[(u(i+1,j)+u(i+1,j+1)) − (u(i,j)+u(i,j+1))] / dy。
    """
    dv_dx = 0.5 * ((v[..., :-1, 1:] + v[..., 1:, 1:])
                   - (v[..., :-1, :-1] + v[..., 1:, :-1])) / float(dx)
    du_dy = 0.5 * ((u[..., 1:, :-1] + u[..., 1:, 1:])
                   - (u[..., :-1, :-1] + u[..., :-1, 1:])) / float(dy)
    return dv_dx - du_dy


# 径向分箱索引缓存:key (H', W', nbins) -> CPU LongTensor(H'*W_half),末桶为哨兵。
_RADIAL_BIN_CACHE = {}


def _radial_bin_indices(H, W, nbins, device):
    """按 (H',W',nbins) 缓存的箱索引;半径单位 cycles/pixel,对数分箱 [1/max, 0.5]。

    返回长度 H'*W_half 的 LongTensor,值为 0..nbins-1(有效箱)或 nbins(范围外哨兵)。
    """
    key = (int(H), int(W), int(nbins))
    idx = _RADIAL_BIN_CACHE.get(key)
    if idx is None:
        ky = np.fft.fftfreq(int(H))          # cycles/pixel
        kx = np.fft.rfftfreq(int(W))
        r = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).reshape(-1)
        lo = 1.0 / max(int(H), int(W))
        edges = np.logspace(np.log10(lo), np.log10(0.5), int(nbins) + 1)
        bin_idx = np.searchsorted(edges, r, side="right") - 1
        valid = (r >= lo) & (r <= 0.5)
        bin_idx = np.where(valid, np.clip(bin_idx, 0, int(nbins) - 1), int(nbins))
        idx = torch.from_numpy(bin_idx.astype(np.int64))
        _RADIAL_BIN_CACHE[key] = idx
    return idx.to(device)


def _radial_log_spectrum_one(f, nbins, b_idx):
    """单个分量 (B,L,H',W') 的径向 log 谱 -> (B,L,nbins)。"""
    B, L, H, W = f.shape
    F = torch.fft.rfft2(f, dim=(-2, -1))
    P = (F.real ** 2 + F.imag ** 2) / float(H * W) ** 2       # |F|²/(H'W')²
    P = P.reshape(B * L, -1)
    npts = P.shape[-1]
    num = P.new_zeros(B * L, int(nbins) + 1).scatter_add_(
        -1, b_idx[None, :].expand(B * L, npts), P)
    cnt = torch.bincount(b_idx, minlength=int(nbins) + 1).to(P.dtype)
    P_bin = num / cnt.clamp(min=1.0)                          # 空箱功率记 0
    return torch.log(P_bin[:, :int(nbins)] + 1e-12).reshape(B, L, int(nbins))


def radial_log_spectra(u, v, nbins=32):
    """逐 (样本,层,分量) 径向平均 log 功率谱,返回 (B,L,2,nbins)。

    输入 u、v 为 destagger 后、已裁到公共 (H-1,W-1) 的质量点场 (B,L,H',W')。
    功率 = |rfft2|²/(H'W')²;半径 sqrt((ky/H')²+(kx/W')²)(cycles/pixel)按
    对数间隔分 nbins 箱,范围 [1/max(H',W'), 0.5],取箱内均值;最后 log(P+1e-12)。
    分量维顺序为 [u, v]。箱索引按形状缓存(模块级 dict),空箱功率记 0
    (log 后为 log(1e-12),两侧相同故不产生梯度)。
    """
    if u.shape != v.shape:
        raise ValueError(f"radial_log_spectra: u {tuple(u.shape)} / v {tuple(v.shape)} 形状不一致")
    H, W = u.shape[-2], u.shape[-1]
    b_idx = _radial_bin_indices(H, W, nbins, u.device)
    su = _radial_log_spectrum_one(u, nbins, b_idx)
    sv = _radial_log_spectrum_one(v, nbins, b_idx)
    return torch.stack([su, sv], dim=2)


def windspeed_levels(u, v, u10, v10, levels):
    """逐样本风速场 (B, len(levels)+1, H-1, W-1)。

    levels 里的层用去交错后的水平风速 sqrt(u_m²+v_m²)(公共质量网格);
    最后一个"层"是 10 m 风速 sqrt(u10²+v10²)(u10/v10 为质量点代理,
    画布形状 (B,H,W),裁到 (H-1,W-1) 丢掉虚行/列)。
    """
    L = u.shape[1]
    li = [int(l) for l in levels]
    if len(li) == 0:
        raise ValueError("windspeed_levels: levels 不能为空")
    if min(li) < 0 or max(li) >= L:
        raise ValueError(f"windspeed_levels: levels {li} 超出层范围 [0,{L})")
    um, vm = _destagger_uv(u, v)
    spd = torch.sqrt(um ** 2 + vm ** 2)                       # (B,L,H-1,W-1)
    sel = spd[:, li]
    spd10 = torch.sqrt(u10[..., :-1, :-1] ** 2 + v10[..., :-1, :-1] ** 2)[:, None]
    return torch.cat([sel, spd10], dim=1)


# ---------------------------------------------------------------------------
# 本地合成自检:python -m src.dl_model.si_follmer.physics_canvas
# ---------------------------------------------------------------------------
def _numpy_truth_divergence(rho, u, v, w, dx, dz):
    """70_truth_diagnostics.divergence_residual 内部差分的 numpy 复刻(仅测试用)。

    rho (nz,ny,nx)、u (nz,ny,nx+1)、v (nz,ny+1,nx)、w (nz+1,ny,nx)、dz (nz,)。
    返回内部体元 (nz-2, ny-2, nx-2) 的 ∇·(ρu)。
    """
    nz = rho.shape[0]
    rho_u = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])
    du = (rho_u[:, :, 1:] * u[:, :, 2:-1] - rho_u[:, :, :-1] * u[:, :, 1:-2]) / dx
    rho_v = 0.5 * (rho[:, :-1, :] + rho[:, 1:, :])
    dv = (rho_v[:, 1:, :] * v[:, 2:-1, :] - rho_v[:, :-1, :] * v[:, 1:-2, :]) / dx
    rho_w = 0.5 * (rho[:-1] + rho[1:])
    dw = (rho_w[1:] * w[2:nz] - rho_w[:-1] * w[1:nz - 1]) / np.maximum(dz[1:-1], 1e-3)[:, None, None]
    return du[1:-1, 1:-1, :] + dv[1:-1, :, 1:-1] + dw[:, 1:-1, 1:-1]


def _place(x):
    """(..., ny, nx) -> (..., 100, 121) 边缘复制(复刻 CanvasStatics.place)。"""
    a = np.asarray(x, dtype=np.float64)
    out = np.empty(a.shape[:-2] + (100, 121), dtype=np.float64)
    h, w = a.shape[-2], a.shape[-1]
    out[..., :h, :w] = a
    if h < 100:
        out[..., h:, :] = out[..., h - 1:h, :]
    if w < 121:
        out[..., :, w:] = out[..., :, w - 1:w]
    return out


def _self_test():
    torch.manual_seed(0)
    np.random.seed(0)
    dx, L, H, W = 1000.0, 4, 20, 24
    ok = True

    def check(name, got, want, tol=1e-6, rel=False):
        nonlocal ok
        if torch.is_tensor(got):
            got = got.detach().numpy()
        diff = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
        base = float(np.max(np.abs(np.asarray(want)))) if rel else 1.0
        good = diff <= tol * max(base, 1.0)
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {name}: max|Δ|={diff:.3e}"
              f"(tol={tol:g}{' rel' if rel else ''})")

    print("① 散度解析场")
    # 常数场
    u = torch.full((1, L, H, W), 3.0)
    v = torch.full((1, L, H, W), -2.0)
    w = torch.full((1, L + 1, H, W), 0.5)
    rho1 = torch.ones(1, L, H, W)
    dz = torch.full((L,), 25.0)
    d = divergence_rho_u(u, v, w, rho1, dx, dz)
    check("常数场 div≈0", d, np.zeros_like(d.numpy()), tol=1e-12)
    # 线性场 u=a·x, v=b·y(ρ=1)-> div = a+b(所有点,含边缘单侧)
    a, b = 2.5e-4, -1.5e-4
    jj = torch.arange(W).view(1, 1, 1, W).float()
    ii = torch.arange(H).view(1, 1, H, 1).float()
    u = a * jj * dx + torch.zeros(1, L, H, W)
    v = b * ii * dx + torch.zeros(1, L, H, W)
    d = divergence_rho_u(u, v, torch.zeros(1, L + 1, H, W), rho1, dx, dz)
    check("线性场 div=a+b", d, np.full(d.shape, a + b))
    # 指数密度廓形 ρ(z):u=a·x, v=b·y, w=0 -> div = ρ_k·(a+b)
    Hs = 800.0
    zk = torch.arange(L).float() * dz
    rho_k = torch.exp(-zk / Hs).view(1, L, 1, 1).expand(1, L, H, W).clone()
    d = divergence_rho_u(u, v, torch.zeros(1, L + 1, H, W), rho_k, dx, dz)
    want = (rho_k * (a + b))[0, :, :H - 1, :W - 1].numpy()
    check("指数 ρ(z) div=ρ_k(a+b)", d, want, tol=1e-6)

    print("② 与真值侧 70_truth_diagnostics 差分公式对拍(同一合成场)")
    nz, ny, nx = L + 2, 12, 15
    dz3 = np.full(nz, 25.0)
    z_mass = np.cumsum(dz3) - dz3 / 2.0
    rho_np = 1.15 * np.exp(-z_mass / 900.0)[:, None, None] * (
        1.0 + 0.05 * np.cos(np.linspace(0, 2 * np.pi, ny)[None, :, None]))
    rho_np = np.broadcast_to(rho_np, (nz, ny, nx)).copy()
    kk = np.arange(nz)[:, None, None]
    yy = np.arange(ny + 1)[None, :, None]
    xx = np.arange(nx + 1)[None, None, :]
    u_np = (2.0 + 0.3 * kk + 0.5 * np.sin(2 * np.pi * xx / nx)) * np.ones((nz, ny, 1))
    v_np = (1.0 - 0.2 * kk + 0.4 * np.cos(2 * np.pi * yy / ny)) * np.ones((nz, 1, nx))
    w_np = 0.05 * np.cos(2 * np.pi * np.arange(nz + 1) / (nz + 1))[:, None, None] \
        * np.ones((nz + 1, ny, nx))
    div_ref = _numpy_truth_divergence(rho_np, u_np, v_np, w_np, dx, dz3)

    y = torch.zeros(1, 3 * nz + 3, 100, 121)
    y[0, 0:nz, :ny, :nx + 1] = torch.from_numpy(u_np)
    y[0, nz:2 * nz, :ny + 1, :nx] = torch.from_numpy(v_np)
    y[0, 2 * nz:3 * nz + 1, :ny, :nx] = torch.from_numpy(w_np)
    y = torch.from_numpy(_place(y[0].numpy())[None]).float()
    u_c, v_c, w_c, _, _ = split_canvas_state(y, nz)
    rho_c = torch.from_numpy(_place(rho_np)[None]).float()
    div_t = divergence_rho_u(u_c, v_c, w_c, rho_c, dx, torch.from_numpy(dz3).float())
    got = div_t[0, 1:nz - 1, 1:ny - 1, 1:nx - 1].numpy()
    check("torch vs 真值差分公式", got, div_ref, tol=1e-5, rel=True)

    print("③ 涡度解析场")
    Om = 3e-4                                # 刚体旋转角速度
    u = -Om * ii * dx + torch.zeros(1, L, H, W)
    v = Om * jj * dx + torch.zeros(1, L, H, W)
    z = vorticity(u, v, dx, dx)
    check("刚体旋转 ζ=2Ω", z, np.full(z.shape, 2 * Om), tol=1e-9)
    z0 = 5e-4
    u = torch.zeros(1, L, H, W)
    v = z0 * jj * dx + torch.zeros(1, L, H, W)
    z = vorticity(u, v, dx, dx)
    check("常数涡度 ζ=ζ0", z, np.full(z.shape, z0), tol=1e-9)

    print("④ 径向 log 谱")
    field = torch.randn(2, L, H - 1, W - 1)
    s1 = radial_log_spectra(field, field * 3.0 + 1.0)
    s2 = radial_log_spectra(field, field * 3.0 + 1.0)
    check("同一场谱差=0", s1 - s2, np.zeros_like(s1.numpy()), tol=1e-12)
    noisy = field + 0.5 * torch.randn_like(field)
    s3 = radial_log_spectra(noisy, noisy)
    d = float(((s1 - s3) ** 2).mean())
    print(f"  [{'OK' if d > 0 else 'FAIL'}] 加噪后谱差>0: mean(ΔlogP)²={d:.4f}")
    ok = ok and d > 0
    print(f"  形状 {tuple(s1.shape)}(期望 (2,{L},2,32)),log 谱范围 "
          f"[{float(s1.min()):.2f},{float(s1.max()):.2f}]")

    print("⑤ 极值项(风速 max/min 结构差)")
    c = 5.0
    u = torch.full((1, L, H, W), c)
    v = torch.zeros(1, L, H, W)
    u10 = torch.full((1, H, W), c)
    v10 = torch.zeros(1, H, W)
    lv = list(range(3))
    sp = windspeed_levels(u, v, u10, v10, lv)
    st = windspeed_levels(u, v, u10, v10, lv)
    ext_same = 0.5 * ((sp.amax(dim=(2, 3)) - st.amax(dim=(2, 3))).abs()
                      + (sp.amin(dim=(2, 3)) - st.amin(dim=(2, 3))).abs()).mean(dim=1)
    check("同场极值=0", ext_same, np.zeros(1), tol=1e-12)
    delta = 2.0
    sp2 = windspeed_levels(u + delta, v, u10 + delta, v10, lv)
    ext_shift = 0.5 * ((sp2.amax(dim=(2, 3)) - st.amax(dim=(2, 3))).abs()
                       + (sp2.amin(dim=(2, 3)) - st.amin(dim=(2, 3))).abs()).mean(dim=1)
    check("整体平移极值=|位移|", ext_shift, np.full(1, delta), tol=1e-9)

    print("⑥ estimate_residual")
    t = torch.tensor([0.0, 0.1, 0.4, 0.5, 0.7, 1.0])
    db = (2.0 * t).view(-1, 1, 1, 1)
    r_true = torch.randn(6, 72, 5, 6)
    b = db * r_true
    r_hat, mask = estimate_residual(b, db, min_t=0.5)
    # 样本 0(t=0,db=0)b=0,只对 t>0 的样本验证精确性
    check("r̂=r(精确系数,t>0)", r_hat[1:], r_true[1:], tol=1e-6, rel=True)
    want_mask = (t >= 0.5).tolist()
    print(f"  [{'OK' if mask.tolist() == want_mask else 'FAIL'}] mask={mask.tolist()}"
          f"(期望 {want_mask})")
    ok = ok and mask.tolist() == want_mask

    print("⑦ 差分/散度梯度非零")
    u = torch.randn(1, L, H, W, requires_grad=True)
    v = torch.randn(1, L, H, W, requires_grad=True)
    w = torch.randn(1, L + 1, H, W, requires_grad=True)
    rho = torch.rand(1, L, H, W) + 0.5
    (divergence_rho_u(u, v, w, rho, dx, dz) ** 2).sum().backward()
    g = float(u.grad.abs().sum() + v.grad.abs().sum() + w.grad.abs().sum())
    print(f"  [{'OK' if g > 0 else 'FAIL'}] 散度对 u/v/w 梯度总量={g:.3f}")
    ok = ok and g > 0

    print("\n自检结果:", "全部通过" if ok else "存在失败项")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_test() else 1)
