import argparse
import torch
from torch.utils.data import DataLoader
from src.models.model import CADFaceModel
from src.data.dataset import CADDataset
from src.training.train import train
from src.training.evaluate import evaluate
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAD Face Model")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Path to processed data')
    parser.add_argument('--split_dir', type=str, default='data/splits', help='Path to split files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='outputs/model.pth', help='Path to save model')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load datasets
    train_dataset = CADDataset(data_dir=args.data_dir, split='train')
    val_dataset = CADDataset(data_dir=args.data_dir, split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = CADFaceModel(input_dim=192, output_dim=args.num_classes)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train model
    train(model, train_loader, val_loader, device, args.epochs, args.lr)

    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
