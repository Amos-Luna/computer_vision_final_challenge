from typing import List, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader


def plot_confusion_matrix(
    model: torch.nn.Module, 
    val_loader: DataLoader,
    device: torch.device,
    classes: List[str]
) -> None:
    """
    Generate and plot confusion matrix for model evaluation.
    
    Args:
        model (torch.nn.Module): Trained model
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to run predictions on
        classes (List[str]): List of class names
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def visualize_predictions(
    model: torch.nn.Module,
    val_dataset: Dataset,
    device: torch.device,
    classes: List[str],
    num_samples: int = 10
) -> None:
    """
    Visualize model predictions on random validation samples.
    
    Args:
        model (torch.nn.Module): Trained model
        val_dataset (Dataset): Validation dataset
        device (torch.device): Device to run predictions on
        classes (List[str]): List of class names
        num_samples (int): Number of samples to visualize
    """
    indices = random.sample(range(len(val_dataset)), num_samples)
    rows = 2
    cols = num_samples // rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image, true_label = val_dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        predicted_class = predicted.item()
        true_class = true_label

        img_np = image.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.set_title(f"Pred: {classes[predicted_class]}\nTrue: {classes[true_class]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: List[float], 
    val_losses: List[float],                      
    train_accs: List[float], 
    val_accs: List[float]
) -> None:
    """
    Plot training and validation metrics history.
    
    Args:
        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        train_accs (List[float]): Training accuracies
        val_accs (List[float]): Validation accuracies
    """
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()