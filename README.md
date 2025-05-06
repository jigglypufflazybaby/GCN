# CAD Face Label Prediction

A machine learning system that predicts manufacturing operation labels (e.g., "extrude_1", "extrude_2") for faces in CAD models using geometric features and graph neural networks.

## Features

- Extracts geometric and topological features from CAD models
- Builds graph representations of CAD models with faces as nodes
- Uses Graph Neural Networks to predict face labels
- Provides visualization tools for results analysis

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Open CASCADE Technology (PythonOCC)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/jigglypufflazybaby/cad-face-prediction.git
cd cad-face-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. For development dependencies:
```bash
pip install -e ".[dev]"
```

5. For visualization dependencies:
```bash
pip install -e ".[visualization]"
```

## Usage

### Data Preparation

1. Place your CAD models in the `data/raw/` directory
2. Run the preprocessing script:
```bash
python scripts/preprocess_data.py --input_dir data/raw/ --output_dir data/processed/
```

### Training

```bash
python scripts/train_model.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate_model.py --model_path models/saved_model.pt --test_data data/processed/test/
```

### Visualization

```bash
python scripts/visualize_predictions.py --model_path models/saved_model.pt --cad_file path/to/cad_file.step
```

## Project Structure

- `src/data/`: Data processing modules
- `src/models/`: Neural network models
- `src/training/`: Training and evaluation code
- `src/utils/`: Utility functions
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `scripts/`: Command-line scripts

## Approach

The system uses a multi-modal approach to extract information from CAD models:

1. **Geometric Feature Extraction**: Processes face normals, centers, areas, and OBBs
2. **Boundary Feature Extraction**: Analyzes boundary points using PointNet-inspired architecture
3. **Topological Feature Extraction**: Captures connectivity information between faces
4. **Graph Neural Network**: Integrates all features and captures relationships between faces
5. **Classification**: Predicts operation labels for each face

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PyTorch and PyTorch Geometric
- Uses Open CASCADE Technology for CAD parsing