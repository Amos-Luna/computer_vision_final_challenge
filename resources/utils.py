# resources/data_utils.py
from typing import Dict, List, Tuple, Union, Optional
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report



def map_labels(label: int) -> int:
    """
    Maps original dataset labels to consolidated categories.
    
    Args:
        label (int): Original label from the dataset
            (0,1) -> COVID
            (2,3) -> Lung Opacity
            (4,5) -> Normal
    
    Returns:
        int: Mapped label (0: COVID, 1: Lung Opacity, 2: Normal)
    
    Raises:
        ValueError: If the input label is not in the expected range
    """
    if label in {0, 1}:
        return 0
    elif label in {2, 3}:
        return 1
    elif label in {4, 5}:
        return 2
    else:
        raise ValueError(f"Unexpected label: {label}")


class CovidDataset(Dataset):
    """
    Custom Dataset class for COVID-19 X-ray images.
    
    Attributes:
        dataset: HuggingFace dataset containing the images and labels
        transform: Optional torchvision transforms to be applied to the images
    """
    
    def __init__(self, hf_dataset: Dataset, transform: Optional[object] = None):
        """
        Initialize the COVID Dataset.
        
        Args:
            hf_dataset (Dataset): HuggingFace dataset containing images and labels
            transform (Optional[object]): Transformations to be applied to images
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (transformed_image, mapped_label)
        """
        sample = self.dataset[idx]
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        label = map_labels(sample['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)


def plot_class_distribution(
    train_dataset: Dataset, 
    val_dataset: Dataset, 
    classes: List[str]
) -> None:
    """
    Plot the class distribution for training and validation sets.
    
    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        classes (List[str]): List of class names
    """
    train_labels_mapped = [map_labels(sample['label']) for sample in train_dataset]
    val_labels_mapped = [map_labels(sample['label']) for sample in val_dataset]
    
    train_counts = Counter(train_labels_mapped)
    val_counts = Counter(val_labels_mapped)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(classes, [train_counts[i] for i in range(len(classes))], color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution - Training")

    plt.subplot(1, 2, 2)
    plt.bar(classes, [val_counts[i] for i in range(len(classes))], color='salmon')
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution - Validation")

    plt.tight_layout()
    plt.show()
    

