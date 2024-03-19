from abc import ABC, abstractmethod
from typing import Literal
import torch
from torch_mi.ranges import PercentileShiftRange


class MutualInformationDensityBased(ABC, torch.nn.Module):
    """
    This class represents a mutual information density-based module in PyTorch.
    It computes the mutual information between two input tensors.

    Attributes:
        nBins (int): The number of bins for histogram computation.
        normalization (function): The normalization function to be used.

    Methods:
        computePxy: Computes the joint probability distribution of two input tensors.
        computeEntropies: Computes the entropies of the input tensors.
        computeKlDivMI: Computes the mutual information using Kullback-Leibler divergence.
        forward: Computes the mutual information between two input tensors.

    """

    def __init__(
        self,
        nBins: int,
        normalization: Literal["none", "sum", "joint", "max", "sqrt", "min"] = "none",
        rangeOpts : dict = {"percentile": 0., "gain": 0.}
    ) -> None:
        """
        Initializes a new instance of the MutualInformationDensityBased class.

        Args:
            nBins (int): The number of bins for histogram computation.
            normalization (Literal["none", "sum", "joint", "max", "sqrt", "min"], optional):
                The type of normalization to be applied. Defaults to "none".

        Raises:
            ValueError: If nBins is not a strictly positive integer.

        """
        super().__init__()
        if not isinstance(nBins, int) or nBins <= 0:
            raise ValueError("nBins must be a strictly positive integer")
        self.nBins = nBins

        self.normalization = self._chooseNorm(normalization)
        
        if (
            not isinstance(rangeOpts, dict)
            or not all([k in rangeOpts for k in ["percentile", "gain"]])
        ):
            raise ValueError("""rangeOpts must be a dictionary with at least keys\
'percentile' and 'gain'""")
        self.range = PercentileShiftRange(**rangeOpts)

    def _batchedLinspace(self, x, nBins):
        # X : (B,N) -> (B, nBins)
        minV, maxV = self.range.computeRange(x)
        return (torch.arange(nBins) * (maxV - minV) / (nBins - 1) + minV)
    
    @abstractmethod
    def computePxy(self, x, y):
        """
        Computes the joint probability distribution of two input tensors.

        Args:
            x (torch.Tensor): The first input tensor of size (B,N).
            y (torch.Tensor): The second input tensor (B,N).

        Returns:
            torch.Tensor: The joint probability distribution.

        """

        pass

    @torch.compile
    def computeEntropies(
        self,
        Pxy,
        Px,
        Py,
    ):
        """
        Computes the entropies of the input tensors.

        Args:
            Pxy (torch.Tensor): The joint probability distribution.
            Px (torch.Tensor): The probability distribution of the first input tensor.
            Py (torch.Tensor): The probability distribution of the second input tensor.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the entropies of the input tensors.

        """

        Px_Py = torch.stack([Px[:, 0], Py[:, :, 0]], dim=0)
        return (
            *(-torch.sum((Px_Py * torch.log(Px_Py)).where(Px_Py > 0, 0), dim=2)),
            -torch.sum((Pxy * torch.log(Pxy)).where(Pxy > 0, 0), dim=(1, 2)),
        )

    @torch.compile
    def computeKlDivMI(self, Pxy, Px, Py):
        """
        Computes the mutual information using Kullback-Leibler divergence.

        Args:
            Pxy (torch.Tensor): The joint probability distribution.
            Px (torch.Tensor): The probability distribution of the first input tensor.
            Py (torch.Tensor): The probability distribution of the second input tensor.

        Returns:
            torch.Tensor: The mutual information.

        """

        PxPy = Px * Py
        return (
            torch.nn.functional.kl_div(
                PxPy.log(),
                Pxy,
                reduction="none",
            )
            .where((Pxy > 0) & (PxPy > 0), 0)
            .sum(dim=(1, 2))
        )

    def _chooseNorm(self, norm):
        """
        Chooses the normalization function based on the given normalization type.

        Args:
            norm (Literal["none", "sum", "joint", "max", "sqrt", "min"]): The normalization type.

        Returns:
            function: The normalization function.

        Raises:
            ValueError: If the normalization type is not valid.

        """

        def sumNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
            return 2 * (Hx + Hy - Hxy) / (Hx + Hy)

        def jointNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / Hxy

        def maxNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / torch.max(Hx, Hy)

        def sqrtNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / torch.sqrt(Hx * Hy)

        def minNorm(Pxy, Px, Py):
            Hx, Hy, Hxy = self.computeEntropies(Px, Py, Pxy)
            return (Hx + Hy - Hxy) / torch.min(Hx, Hy)

        match norm:
            case "none":
                return self.computeKlDivMI
            case "sum":
                return sumNorm
            case "joint":
                return jointNorm
            case "max":
                return maxNorm
            case "sqrt":
                return sqrtNorm
            case "min":
                return minNorm
            case _:
                raise ValueError(
                    "Normalization must be one of 'none', 'sum', 'joint', 'max', 'sqrt', 'min'"
                )

    def forward(self, x, y):
        """
        Computes the mutual information between two input tensors.

        Args:
            x (torch.Tensor): The first input tensor (B,C,H,W).
            y (torch.Tensor): The second input tensor (B,C,H,W).

        Returns:
            torch.Tensor: The channel-wise mutual information (B,C).

        Raises:
            AssertionError: If the shapes of x and y are not the same.

        """

        assert x.shape == y.shape, "Tensor x and y must have the same shape"
        Pxy = self.computePxy(
            x.view(x.size(0) * x.size(1), x.size(2) * x.size(3)),
            y.view(y.size(0) * y.size(1), y.size(2) * y.size(3)),
        )
        return (
            self.normalization(
                Pxy, Pxy.sum(dim=1, keepdim=True), Pxy.sum(dim=2, keepdim=True)
            ).reshape(x.size(0), x.size(1))
        ).clamp(
            min=0,
        )
