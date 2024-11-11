# OMTFGnnTrain.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn as nn
import logging

def train(model, optimizer, criterion, loader, device, scaler=None, grad_clip=None, profiler=None):
    """
    Trains the GNN model for one epoch.

    Parameters:
    - model (nn.Module): The GNN model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - criterion (nn.Module): Loss function.
    - loader (DataLoader): DataLoader for training data.
    - device (str): Device to run the training on ('cuda' or 'cpu').
    - scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision training.
    - grad_clip (float, optional): Maximum norm for gradients.
    - profiler (torch.profiler.profile, optional): PyTorch Profiler instance.

    Returns:
    - float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(batch)
                loss = criterion(out, batch.y.argmax(dim=1))
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch)
            loss = criterion(out, batch.y.argmax(dim=1))
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * batch.num_graphs

        # Profiler step
        if profiler:
            profiler.step()

    avg_loss = total_loss / len(loader.dataset)
    logging.info(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, criterion, loader, device):
    """
    Evaluates the GNN model on the validation set.

    Parameters:
    - model (nn.Module): The GNN model to evaluate.
    - criterion (nn.Module): Loss function.
    - loader (DataLoader): DataLoader for validation data.
    - device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
    - tuple: (Average validation loss, validation accuracy)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.argmax(dim=1))
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Parameters:
    - model (nn.Module): The GNN model.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - epoch (int): Current epoch number.
    - path (str): File path to save the checkpoint.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    logging.info(f'Checkpoint saved at epoch {epoch} to {path}')

def load_checkpoint(model, optimizer, path, device):
    """
    Loads the model and optimizer state from a checkpoint file.

    Parameters:
    - model (nn.Module): The GNN model.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - path (str): File path to load the checkpoint from.
    - device (str): Device to load the model onto.

    Returns:
    - int: The epoch number from which training was resumed.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logging.info(f'Checkpoint loaded from {path} at epoch {epoch}')
    return epoch
