from .db import MutualInformationDensityBased
from typing import Literal
import torch


class BinsCountMutualInformation(MutualInformationDensityBased):
    """
    BinsCountMutualInformation class calculates the mutual information between two variables
    using bin counts.

    Args:
        nBins (int): The number of bins to use for histogram calculation.
        normalization (Literal["none", "sum", "joint", "max", "sqrt", "min"], optional):
            The type of normalization to apply to the histogram. Defaults to "none".
        mode (Literal["soft", "discrete"], optional): The mode of histogram calculation. "soft"
            uses a soft histogram calculation, while "discrete" uses a discrete histogram 
            calculation. Defaults to "soft".
        rangeOpts (dict) : Options to define the linspace range for the histogram calculation.

    Attributes:
        nBins (int): The number of bins used for histogram calculation.
        normalization (Literal["none", "sum", "joint", "max", "sqrt", "min"]): 
            The type of normalization applied to the histogram.
        computeHist (function): The function used for histogram calculation based on the mode.
        rangeOpts : Options to define the linspace range for the histogram calculation.

    Methods:
        softHistogram(x, y, binsX, binsY): Calculates the soft histogram between two variables.
        discreteHistogram(x, y, binsX, binsY): Calculates the discrete histogram between two 
            variables.
        normalize(x, dim): Normalizes the input tensor along the specified dimension.
        computePxy(x, y): Computes the joint probability distribution P(x, y) based on the input 
            variables x and y.

    """

    def __init__(
        self,
        nBins: int,
        normalization: Literal["none", "sum", "joint", "max", "sqrt", "min"] = "none",
        mode: Literal["soft", "discrete"] = "soft",
        rangeOpts : dict = {"percentile": 0., "gain": 0.}
    ):
        super().__init__(nBins=nBins, normalization=normalization, rangeOpts=rangeOpts)
        self.computeHist = self.softHistogram if mode == "soft" else self.discreteHistogram

    def softHistogram(self, x, y, binsX, binsY):
        """
        Calculates the soft histogram between two variables.

        Args:
            x (torch.Tensor): The input tensor for variable x (B,N,1).
            y (torch.Tensor): The input tensor for variable y (B,N,1).
            binsX (torch.Tensor): The bins for variable x (B,1, nBins+1).
            binsY (torch.Tensor): The bins for variable y (B,1, nBins+1).

        Returns:
            torch.Tensor: The soft histogram between variables x and y (B, nBins, nBins).

        """
        return (
            (
                (torch.sigmoid(1e6 * (x - binsX).unsqueeze(-2)))  # (B,N,1, nBins+1)
                * (torch.sigmoid(1e6 * (y - binsY).unsqueeze(-1)))  # (B,N, nBins+1, 1)
            )  # (B,N, nBins+1, nBins+1)
            .sum(dim=1)  # (B, nBins+1, nBins+1)
            .diff(dim=-1)  # (B, nBins+1, nBins)
            .diff(dim=-2)  # (B, nBins, nBins)
        )

    def discreteHistogram(self, x, y, binsX, binsY):
        """
        Calculates the discrete histogram between two variables.

        Args:
            x (torch.Tensor): The input tensor for variable x (B,N,1).
            y (torch.Tensor): The input tensor for variable y (B,N,1).
            binsX (torch.Tensor): The bins for variable x (B,1, nBins+1).
            binsY (torch.Tensor): The bins for variable y (B,1, nBins+1).

        Returns:
            torch.Tensor: The discrete histogram between variables x and y.

        """
        return (
            (
                (x >= binsX).unsqueeze(-2)
                & ((x <= (binsX + binsX[: , :, [0, -1]].diff() / self.nBins)).unsqueeze(-2))
                & (y >= binsY).unsqueeze(-1)
                & ((y <= (binsY + binsY[: , :, [0, -1]].diff() / self.nBins)).unsqueeze(-1))
            ).sum(dim=1)
        )[:, :-1, :-1]

    def normalize(self, x, dim):
        """
        Normalizes the input tensor along the specified dimension.

        Args:
            x (torch.Tensor): The input tensor to normalize.
            dim (int): The dimension along which to normalize.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x / x.sum(dim=dim, keepdim=True)

    def computePxy(self, x, y):
        """
        Computes the joint probability distribution P(x, y) based on the input variables x and y.

        Args:
            x (torch.Tensor): The input tensor for variable x (B,N).
            y (torch.Tensor): The input tensor for variable y (B,N).

        Returns:
            torch.Tensor: The joint probability distribution P(x, y).

        """
        return self.normalize(
            self.computeHist(
                x.unsqueeze(-1),
                y.unsqueeze(-1),
                self._batchedLinspace(x, self.nBins + 1).unsqueeze(1),
                self._batchedLinspace(y, self.nBins + 1).unsqueeze(1)
            ),
            (1, 2),
        )
