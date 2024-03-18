from typing import Optional, Tuple, Literal
from .db import MutualInformationDensityBased
import torch


class KdeMutualInformation(MutualInformationDensityBased):
    """
    This class represents a kernel density estimation (KDE) based mutual information computation.

    Args:
        nBins (int): The number of bins to use for discretizing the data.
        bandwidths (Optional[Tuple[float | int, float | int] | float | int], optional):
            The bandwidths to use for KDE.
            If None, the bandwidths will be computed using the Silverman method. Defaults to None.
        normalization (Literal["none", "sum", "joint", "max", "sqrt", "min"], optional):
            The type of normalization to apply.
            Can be one of the following: "none", "sum", "joint", "max", "sqrt", "min".
            Defaults to "none".

    Attributes:
        bandwidths (Optional[Tuple[float | int, float | int] | float | int]):
            The bandwidths used for KDE.
        computeBandwidths (function): The function used to compute the bandwidths.

    Methods:
        _computeBandwidths: Computes the bandwidths based on the given method or user-defined
        values.
        _computeExpResiduals: Computes the exponential residuals for KDE.
        computePxy: Computes the joint probability distribution P(x, y) using KDE.

    Inherits from:
        MutualInformationDensityBased: A base class for mutual information computation based on
        density estimation.
    """

    def __init__(
        self,
        nBins: int,
        bandwidths: Optional[Tuple[float | int, float | int] | float | int] = None,
        normalization: Literal["none", "sum", "joint", "max", "sqrt", "min"] = "none",
    ):
        super().__init__(nBins=nBins, normalization=normalization)
        if bandwidths is not None:
            if isinstance(bandwidths, (float, int)):
                bandwidths = 2 * [bandwidths]
            if (
                not isinstance(bandwidths, [tuple, list])
                or any(bw <= 0 or not isinstance(bw, (float, int)) for bw in bandwidths)
                or len(bandwidths) != 2
            ):
                raise ValueError(
                    """Bandwidths must be a strictly positive number\
or a collection of two strictly positive floats"""
                )
        self.bandwidths = bandwidths
        self.computeBandwidths = self._computeBandwidths()

    def _computeBandwidths(self):
        """
        Computes the bandwidths based on the given method or user-defined values.

        Returns:
            function: The function used to compute the bandwidths.
        """

        def silvermanMethod(x, _):
            # If d=2, Scott and Silverman are equivalent
            return (x.size(1) ** (-1 / 6)) * x.std()

        def userDefinedMethod(_, dim):
            return self.bandwidths[dim]

        return silvermanMethod if self.bandwidths is None else userDefinedMethod

    def _computeExpResiduals(self, x: torch.Tensor, h: float):
        """
        Computes the exponential residuals for KDE.

        Args:
            x (torch.Tensor): The input tensor (B,N).
            h (float): The bandwidth.

        Returns:
            torch.Tensor: The computed exponential residuals (B, N, nBins).
        """
        return (
            -0.5
            * (
                (
                    torch.linspace(*x.aminmax(), self.nBins).unsqueeze(0).unsqueeze(0)
                    - x.unsqueeze(-1)
                )
                / h
            ).pow(2)
        ).exp()

    def computePxy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Computes the joint probability distribution P(x, y) using KDE.

        Args:
            x (torch.Tensor): The input tensor for variable x (B,N).
            y (torch.Tensor): The input tensor for variable y (B,N).

        Returns:
            torch.Tensor: The computed joint probability distribution P(x, y) (B,nBins,nBins).
        """
        expResXY = torch.einsum(
            "bki,bkj -> bij",
            self._computeExpResiduals(x, self.computeBandwidths(x, 0)),
            self._computeExpResiduals(y, self.computeBandwidths(y, 1)),
        )
        return expResXY / (expResXY.sum(dim=(1, 2), keepdim=True))
