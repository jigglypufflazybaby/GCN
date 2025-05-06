import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # For multi-label classification

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1
