# src/training/train.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.config import cfg
from src.data.dataset import CADGraphDataset
from src.models.model import CADOperationGCNModel
from src.training.evaluate import evaluate_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = CADGraphDataset(cfg.train_split, cfg.data_root)
    val_dataset = CADGraphDataset(cfg.val_split, cfg.data_root)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = CADOperationGCNModel(cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = torch.nn.BCELoss()

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            face_feats = batch['face_features'].to(device)         # [num_faces, in_dim]
            boundary_pts = batch['boundary_points'].to(device)     # [num_faces, N, 3]
            adj = batch['adjacency'].to(device)                    # [num_faces, num_faces]
            labels = batch['labels'].to(device).float()            # [num_faces, num_classes]

            optimizer.zero_grad()
            outputs = model(face_feats, boundary_pts, adj)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

        if (epoch + 1) % cfg.eval_interval == 0:
            evaluate_model(model, val_loader, device)

        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    train()
