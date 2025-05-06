import torch
import torch.nn as nn

class FaceFeatureProjector(nn.Module):
    def __init__(self, in_dim: int = 19, out_dim: int = 128):
        """
        Projects raw face features to higher-dimensional space.
        Args:
            in_dim: Number of input features (default 19).
            out_dim: Output embedding size.
        """
        super(FaceFeatureProjector, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.projection(x)  # Output: [num_faces, out_dim]
