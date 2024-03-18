import torch


class PercentileShiftRange:
    """
    This class represents a range calculator for mutual information density-based modules in
    PyTorch.

    Methods:
        computeRange: Computes the range of the input tensor.

    Attributes:
        percentile (float): The percentile value for computing the range.
        gain (float): The gain value for shifting the range.

    """

    def __init__(self, percentile: float = 0.0, gain: float = 0.0):
        """
        Initializes a new instance of the PercentileShiftRange class.

        Args:
            percentile (float): The percentile value for computing the range.
            gain (float): The gain value for shifting the range.

        Raises:
            ValueError: If percentile is not in the range [0, 1].
            ValueError: If gain is not in the range [0, 1].

        """
        if percentile < 0 or percentile > 1:
            raise ValueError("Percentile must be in the range [0, 1]")
        self.percentile = percentile
        if gain < 0 or gain > 1:
            raise ValueError("Shift must be in the range [0, 1]")
        self.gain = gain

    def computeRange(self, x):
        """
        Computes the range of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the lower and upper bounds of the range.

        """
        lowerB, upperB = torch.quantile(x, [self.percentile, 1 - self.percentile])
        return lowerB - self.gain * lowerB.abs(), upperB + self.gain * upperB.abs()
