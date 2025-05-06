import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractors import GeometricFeatureMLP, BoundaryPointNet, TopologyGCN
from .gnn import GCNBlock

class FaceOperationGCN(nn.Module):
    def __init__(self, num_classes, geom_in_dim=13, boundary_dim=3, topo_dim=1):
        super(FaceOperationGCN, self).__init__()

        # Feature extractors
        self.geom_mlp = GeometricFeatureMLP(input_dim=geom_in_dim, hidden_dims=[32, 64])
        self.boundary_net = BoundaryPointNet(point_dim=boundary_dim, output_dim=64)
        self.topo_gcn = TopologyGCN(input_dim=topo_dim, output_dim=64)

        # GCN Stack
        self.gcn1 = GCNBlock(in_dim=192, out_dim=256)
        self.gcn2 = GCNBlock(in_dim=256, out_dim=256)
        self.gcn3 = GCNBlock(in_dim=256, out_dim=128)

        # Global + Local context (assume pooling is done externally if needed)
        self.fc1 = nn.Linear(128 + 64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, geom_feat, boundary_pts, topo_feat, face_edge_index, global_context=None, local_context=None):
        """
        Args:
            geom_feat:       [num_faces, 13]         => (normal[3], center[3], area[1], obb_dims[3], obb_orient[3])
            boundary_pts:    [num_faces, N, 3]       => boundary points per face
            topo_feat:       [num_faces, 1]          => optional: num_adjacent_faces, or angles
            face_edge_index: [2, num_edges]          => face adjacency graph
            global_context:  [num_faces, 128]        => optional (graph pooled)
            local_context:   [num_faces, 64]         => optional (neighborhood pooled)
        Returns:
            logits:          [num_faces, num_classes]
        """
        # Extract features
        geom_out = self.geom_mlp(geom_feat)                 # → [num_faces, 64]
        boundary_out = self.boundary_net(boundary_pts)      # → [num_faces, 64]
        topo_out = self.topo_gcn(topo_feat, face_edge_index)# → [num_faces, 64]

        x = torch.cat([geom_out, boundary_out, topo_out], dim=1)  # → [num_faces, 192]

        # GCN Stack
        x = self.gcn1(x, face_edge_index)  # → [num_faces, 256]
        x = self.gcn2(x, face_edge_index)  # → [num_faces, 256]
        x = self.gcn3(x, face_edge_index)  # → [num_faces, 128]

        # Context integration
        if global_context is not None and local_context is not None:
            x = torch.cat([x, local_context], dim=1)  # → [num_faces, 128 + 64]
        else:
            # fallback if no pooling implemented
            x = torch.cat([x, torch.zeros_like(x[:, :64])], dim=1)

        # Classifier
        x = F.relu(self.fc1(x))            # → [num_faces, 128]
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))            # → [num_faces, 64]
        x = self.fc3(x)                    # → [num_faces, num_classes]
        return torch.sigmoid(x)           # For multi-label classification
