import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Dict, Tuple
import json

from .cad_parser import load_cad_sample
from .feature_extraction import build_face_features

class CADFaceDataset(Dataset):
    def __init__(self, data_dir: str, label_map: Dict[str, int], transform=None):
        """
        Args:
            data_dir: Directory with JSON files.
            label_map: Mapping from operation string (e.g. "extrude1") to integer class.
            transform: Optional transform function.
        """
        self.data_dir = data_dir
        self.filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        faces, edges = load_cad_sample(file_path)

        # Build feature matrix
        x = build_face_features(faces)                    # [num_faces, 19]
        x = torch.tensor(x, dtype=torch.float)

        # Build edge_index
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]

        # Build multi-label tensor [num_faces, N_labels]
        y = torch.zeros((len(faces), len(self.label_map)), dtype=torch.float)
        for i, face in enumerate(faces):
            for label in face['labels']:
                if label in self.label_map:
                    y[i, self.label_map[label]] = 1.0

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.transform:
            data = self.transform(data)

        return data
