# src/training/config.py

from easydict import EasyDict

cfg = EasyDict()

# Paths
cfg.data_root = 'data/processed'
cfg.train_split = 'data/splits/train.json'
cfg.val_split = 'data/splits/val.json'
cfg.test_split = 'data/splits/test.json'

# Training settings
cfg.batch_size = 1                  # One model per sample due to variable size
cfg.num_epochs = 100
cfg.learning_rate = 1e-3
cfg.weight_decay = 5e-4

# Model
cfg.input_dims = {
    'geometric': 13,               # 3+3+1+3+3 (normal + center + area + OBB dims + OBB quat)
    'boundary': 3,                 # Nx3 points
    'topological': 1               # Only needs adjacency
}
cfg.hidden_dim = 64
cfg.gnn_hidden_dim = 256
cfg.gnn_output_dim = 128
cfg.global_pooling = 'mean'        # or 'max'

# Output
cfg.num_classes = 10               # Update based on number of operations (extrude, fillet, etc.)

# Logging
cfg.log_interval = 10
cfg.eval_interval = 1
