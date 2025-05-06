import numpy as np
from typing import List, Dict, Tuple

def normalize_vector(vec: List[float]) -> List[float]:
    norm = np.linalg.norm(vec)
    return (np.array(vec) / norm).tolist() if norm != 0 else vec

def flatten_obb(obb: List[List[float]]) -> List[float]:
    return [val for row in obb for val in row]

def build_face_features(faces: List[Dict]) -> np.ndarray:
    """
    Constructs a [num_faces, num_features] feature matrix.

    Features per face:
        - normal (3D) → normalized
        - center (3D)
        - flattened OBB (3x3 = 9D)
        - avg_point_distance (1D)
        - point_density (1D)
        - num_boundary_points (1D)
        - mean angle with neighbors (1D)

    Total: 3 + 3 + 9 + 1 + 1 + 1 + 1 = 19 features per face
    """
    features = []
    for face in faces:
        normal = normalize_vector(face['normal'])                     # 3D
        center = face['center']                                      # 3D
        obb = flatten_obb(face['obb'])                                # 9D
        avg_distance = [face['avg_distance']]                         # 1D
        density = [face['density']]                                   # 1D
        num_boundary = [face['num_boundary_points']]                  # 1D
        mean_angle = [np.mean(face['angle_neighbors'])]               # 1D

        feature_vector = normal + center + obb + avg_distance + density + num_boundary + mean_angle
        features.append(feature_vector)

    return np.array(features, dtype=np.float32)
