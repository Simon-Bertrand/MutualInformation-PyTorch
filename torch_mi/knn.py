import torch


class KnnMutualInformation(torch.nn.Module):
    """
    KnnMutualInformation is a PyTorch module that calculates the K-nearest neighbors
    mutual information between two tensors.

    Args:
        nNeighbours (int): The number of nearest neighbors to consider.

    Raises:
        ValueError: If nNeighbours is not a strictly positive integer.

    Attributes:
        nNeighbours (int): The number of nearest neighbors to consider.

    Methods:
        forward(x, y): Calculates the K-nearest neighbors mutual information between tensors
        x and y.

    """

    def __init__(
        self,
        nNeighbours: int,
    ):
        super().__init__()
        if not isinstance(nNeighbours, int) or nNeighbours <= 0:
            raise ValueError("nNeighbours must be a strictly positive integer")
        self.nNeighbours = nNeighbours

    def forward(self, x, y):
        """
        Calculates the K-nearest neighbors mutual information between tensors x and y.

        Args:
            x (torch.Tensor): The input tensor x (B,C,H,W).
            y (torch.Tensor): The input tensor y (B,C,H,W).

        Returns:
            torch.Tensor: The channel-wise mutual information between x and y (B,C).

        Raises:
            AssertionError: If x and y do not have the same shape.

        """
        assert x.shape == y.shape, "Tensor x and y must have the same shape"
        BC, HW = x.size(0) * x.size(1), x.size(2) * x.size(3)
        zNormXy = torch.cdist(*2 * [torch.stack([d.view(BC, HW, 1) for d in (x, y)])], p=1)
        # We can't use p=torch.inf as we need individually the distances for each dimension
        eps = (
            zNormXy.max(dim=0)
            .values.topk(self.nNeighbours + 1, dim=-1, largest=False)
            .values[:, :, -1:]
        )
        return (
            torch.digamma(torch.tensor([self.nNeighbours, HW])).sum()
            - (torch.digamma((zNormXy < eps.unsqueeze(0)).sum(dim=-1)).sum(dim=0))
            .mean(dim=-1)
            .reshape(x.size(0), x.size(1))
        ).clamp(
            min=0,
        )
