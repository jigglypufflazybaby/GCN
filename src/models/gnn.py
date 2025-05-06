import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, residual=False):
        """
        A basic GCN block that uses graph convolution, batch normalization, ReLU, and dropout with optional residual connections.
        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            dropout: Dropout rate.
            residual: Whether to add residual connections.
        """
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.bn = BatchNorm(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and in_dim == out_dim

    def forward(self, x, edge_index):
        """
        Forward pass for the GCN block.
        Args:
            x: Node features [num_faces, in_dim].
            edge_index: Graph connectivity [2, num_edges].
        Returns:
            x: Updated node features [num_faces, out_dim].
        """
        identity = x if self.residual else None
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.residual:
            x = x + identity
        return x


class TopologyGCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        This module processes the topological features of faces via GCN.
        Args:
            input_dim: The input feature dimension.
            output_dim: The output feature dimension after GCN processing.
        """
        super(TopologyGCN, self).__init__()
        self.gcn = GCNBlock(in_dim=input_dim, out_dim=output_dim)

    def forward(self, topo_feat, face_edge_index):
        """
        Forward pass for the TopologyGCN module.
        Args:
            topo_feat: Topological feature for each face.
            face_edge_index: Graph connectivity [2, num_edges].
        Returns:
            x: Output features after GCN processing [num_faces, output_dim].
        """
        return self.gcn(topo_feat, face_edge_index)
