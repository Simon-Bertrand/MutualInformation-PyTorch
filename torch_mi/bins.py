from .db import MutualInformationDensityBased
from typing import Literal
import torch


class BinsCountMutualInformation(MutualInformationDensityBased):
    def __init__(
        self,
        nBins: int,
        normalization: Literal["none", "sum", "joint", "max", "sqrt", "min"] = "none",
        mode: Literal["soft", "discrete"] = "soft",
    ):
        super().__init__(nBins=nBins, normalization=normalization)
        self.computeHist = self.softHistogram if mode == "soft" else self.discreteHistogram

    def softHistogram(self, x, y, binsX, binsY):
        return (
            (
                (torch.sigmoid(1e6 * (x - binsX).unsqueeze(-2)))
                * (torch.sigmoid(1e6 * (y - binsY).unsqueeze(-1)))
            )
            .sum(dim=1)
            .diff(dim=-1)
            .diff(dim=-2)
        )

    def discreteHistogram(self, x, y, binsX, binsY):
        return (
            (
                (x >= binsX).unsqueeze(-2)
                & ((x <= (binsX + binsX[[0, -1]].diff() / self.nBins)).unsqueeze(-2))
                & (y >= binsY).unsqueeze(-1)
                & ((y <= (binsY + binsY[[0, -1]].diff() / self.nBins)).unsqueeze(-1))
            ).sum(dim=1)
        )[:, :-1, :-1]

    def normalize(self, x, dim):
        return x / x.sum(dim=dim, keepdim=True)

    def computePxy(self, x, y):
        return self.normalize(
            self.computeHist(
                x.unsqueeze(-1),
                y.unsqueeze(-1),
                torch.linspace(*x.aminmax(), self.nBins + 1),
                torch.linspace(*y.aminmax(), self.nBins + 1),
            ),
            (1, 2),
        )
