import json
import os
from src.data.feature_extraction import feature_extractor

def preprocess_raw_data(input_dir, output_dir):
    """Preprocess raw CAD files into features."""
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            with open(os.path.join(input_dir, file), 'r') as f:
                data = json.load(f)
            
            # Extract features using the feature extractor
            features = feature_extractor(data)
            
            # Save processed features
            output_file = os.path.join(output_dir, file)
            with open(output_file, 'w') as f:
                json.dump(features, f)

if __name__ == "__main__":
    preprocess_raw_data('data/raw', 'data/processed')
