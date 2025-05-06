import json
import os
from typing import List, Dict, Tuple, Any

def load_json(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def parse_faces(json_data: Dict) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    """
    Parses face information and face adjacency from a CAD JSON object.

    Returns:
        faces: List of face attributes (dicts)
        edges: List of (face_i, face_j) tuples representing connectivity
    """
    faces = []
    id_to_index = {}
    
    for idx, face in enumerate(json_data['faces']):
        fid = face['face_id']
        id_to_index[fid] = idx
        faces.append({
            'face_id': fid,
            'center': face['center'],
            'normal': face['normal'],
            'obb': face['obb'],
            'boundary': face['boundary_points'],
            'angle_neighbors': face['angle_with_neighbors'],  # list of floats
            'avg_distance': face['avg_point_distance'],
            'density': face['point_density'],
            'num_boundary_points': len(face['boundary_points']),
            'labels': face['labels']  # List[str], e.g., ["extrude1", "fillet1"]
        })

    # Parse face adjacency
    edges = []
    for conn in json_data['connectivity']:
        src, tgt = id_to_index[conn[0]], id_to_index[conn[1]]
        edges.append((src, tgt))

    return faces, edges

def load_cad_sample(filepath: str) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    json_data = load_json(filepath)
    return parse_faces(json_data)
