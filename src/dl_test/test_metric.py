import sys
from logging import getLogger

import torch
from torch import nn

from src.dl_test.ssim2d import SSIM2D

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


class SSIM2DLoss(nn.Module):
    def __init__(
        self,
        max_value: float,
        scale: float,
        bias: float,
        window_size: int = 11,
        sigma: float = 1.5,
        use_gauss: bool = False,
        offset: float = 0.0,
    ):
        super().__init__()
        self.ssim = SSIM2D(
            scale=scale,
            bias=bias,
            window_size=window_size,
            sigma=sigma,
            size_average=False,
            max_value=max_value,
            use_gauss=use_gauss,
            offset=offset,
        )

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        #
        assert predicts.ndim == targets.ndim == 4  # dims: batch, channel, y, x

        ssims = self.ssim(img1=predicts, img2=targets)

        # mean along channel, y, and x
        return 1.0 - torch.mean(ssims, dim=(1, 2, 3))


class L1Loss(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
        #
        assert predicts.ndim == targets.ndim == 4  # dims: batch, channel, y, x

        diffs = torch.abs(predicts - targets)
        return torch.mean(diffs, dim=(1, 2, 3)) * self.scale


class RMSE(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
        #
        assert predicts.ndim == targets.ndim == 4  # dims: batch, channel, y, x

        diffs = predicts - targets
        mse = torch.mean(diffs**2, dim=(1, 2, 3))

        return torch.sqrt(mse) * self.scale


def rank_histogram(preds: torch.Tensor, obs: torch.Tensor):
    assert preds.ndim == 4  # dims: batch (=n_case), ensemble (=n_member), y, x
    assert obs.ndim == 3  # dims: batch, y, x

    N_case, N_member, H, W = preds.shape
    assert obs.shape == (N_case, H, W)

    ens = preds.permute(0, 2, 3, 1).reshape(-1, N_member)
    truth = obs.reshape(-1, 1)

    ranks = (ens < truth).sum(dim=1)
    assert ranks.shape == (N_case * H * W,)
    hist = torch.bincount(ranks, minlength=N_member + 1)
    # shape: (N_member+1,), this is a histogram for ranks

    return hist, ranks


def spread_skill_ratio(preds: torch.Tensor, obs: torch.Tensor):
    assert preds.ndim == 4  # dims: batch (=n_case), ensemble (=n_member), y, x
    assert obs.ndim == 3  # dims: batch, y, x

    N_case, N_member, H, W = preds.shape
    assert obs.shape == (N_case, H, W)

    ens_mean = preds.mean(dim=1)  # (N_case, H, W)
    ens_std = preds.std(dim=1, unbiased=False)
    assert ens_mean.shape == ens_std.shape == (N_case, H, W)

    rmse = torch.sqrt(((ens_mean - obs) ** 2).mean())
    spread = ens_std.mean()

    return (spread / rmse).item(), rmse, spread


def crps_ensemble(preds: torch.Tensor, obs: torch.Tensor):
    assert preds.ndim == 4  # dims: batch (=n_case), ensemble (=n_member), y, x
    assert obs.ndim == 3  # dims: batch, y, x

    N_case, N_member, H, W = preds.shape
    assert obs.shape == (N_case, H, W)

    ens = preds.view(N_case, N_member, -1)  # dims: n_case, n_mmember, p (=n_x*n_y)
    truth = obs.view(N_case, -1)  # dims: n_case, p

    term1 = (ens - truth.unsqueeze(1)).abs().mean(dim=1)  # (n_case, p)

    diffs = ens.unsqueeze(2) - ens.unsqueeze(1)  # (n_case, n_member, n_member, p)
    assert diffs.shape == (N_case, N_member, N_member, H * W)
    term2 = diffs.abs().mean(dim=(1, 2)) * 0.5  # (n_case, p)

    assert term1.shape == term2.shape == (N_case, H * W)

    crps = term1 - term2  # (n_case, p)
    return crps.view(N_case, H, W)
