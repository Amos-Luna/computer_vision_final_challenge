from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import copy
import timm
import torch
import torch.nn as nn
import copy
import os


class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting.
    
    Attributes:
        patience (int): Number of epochs to wait before stopping
        best_loss (float): Best validation loss achieved
        counter (int): Counter for patience
        best_model_wts (Optional[Dict]): Best model weights
    """
    
    def __init__(self, patience: int = 7):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_wts = None

    def step(
        self, 
        val_loss: float, 
        model: nn.Module
    ) -> bool:
        """
        Perform a step of early stopping.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Current model
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def create_xception_model(
    num_classes: int = 3, 
    dropout_rate: float = 0.7
) -> nn.Module:
    """
    Create and initialize the Xception model.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        nn.Module: Initialized model
    """
    model = timm.create_model('xception', pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    return model


def create_resnet50_model(
    num_classes: int = 3, 
    dropout_rate: float = 0.7
) -> nn.Module:
    #Implement
    pass


def create_efficientnet_v2_s_model(
    num_classes: int = 3, 
    dropout_rate: float = 0.7
) -> nn.Module:
    #Implement
    pass


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int = 30,
    save_dir: str = "./models",
    model_prefix: str = "xception_model"
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model with early stopping and model checkpointing.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss criterion
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to train on
        num_epochs (int): Maximum number of epochs to train
        save_dir (str): Directory to save model checkpoints
        model_prefix (str): Prefix for saved model files
        
    Returns:
        Tuple[nn.Module, Dict[str, List[float]]]: Trained model and training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    early_stopping = EarlyStopping(patience=7)
    
    print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
    print("------------------------------------------------------")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss, running_corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        t_loss = running_loss / len(train_loader.dataset)
        t_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_loss, running_corrects = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        v_loss = running_loss / len(val_loader.dataset)
        v_acc = running_corrects.double() / len(val_loader.dataset)

        # Update learning rate
        scheduler.step(v_loss)
        
        # Save metrics
        train_losses.append(t_loss)
        train_accs.append(t_acc.item())
        val_losses.append(v_loss)
        val_accs.append(v_acc.item())

        print(f"{epoch+1:5d} | {t_loss:10.4f} | {t_acc:9.4f} | {v_loss:8.4f} | {v_acc:7.4f}")

        # Save best model
        if v_acc > best_acc:
            best_acc = v_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = os.path.join(save_dir, f"{model_prefix}_epoch_{epoch+1}_4to_intent.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

        # Early stopping check
        if early_stopping.step(v_loss, model):
            print("Early stopping activated. Ending training.")
            break

    print("------------------------------------------------------")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return training history
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    
    return model, history