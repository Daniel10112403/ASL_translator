# install required libraries
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch

from typing import List
import csv

# Define a custom dataset class for sign language classification
class lengua_senas_dataset(Dataset):
    '''Sign language classification dataset.

    Utility for loading Sign Language dataset into PyTorch.

    Each sample is 1 x 1 x 28 x 28, and each label is a scalar. 
    '''

    # Static method to get label mapping
    @staticmethod
    def get_label_mapping():
        """
        We map all labels to [0, 23]. This mapping from dataset labels [0, 23]
        to letter indices [0, 25] is returned below.
        """
        mapping = list(range(25))
        mapping.pop(9)  # Remove the label 9 from the mapping
        return mapping

    # Static method to read label samples from a CSV file
    @staticmethod
    def read_label_samples_from_csv(path: str):
        """
        Assumes first column in CSV is the label and subsequent 28^2 values
        are image pixel values 0-255.
        """
        mapping = lengua_senas_dataset.get_label_mapping()
        labels, samples = [], []
        with open(path) as f:
            _ = next(f)  # skip header
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))  # Map label to index
                samples.append(list(map(int, line[1:])))  # Convert pixel values to integers
        return labels, samples

    # Initialize the dataset
    def __init__(self,
            path: str="data_2/sign_mnist_train.csv",
            mean: List[float]=[0.485],
            std: List[float]=[0.229]):
        """
        Args:
            path: Path to `.csv` file containing `label`, `pixel0`, `pixel1`...
        """
        labels, samples = lengua_senas_dataset.read_label_samples_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))  # Reshape samples
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))  # Reshape labels

        self._mean = mean  # Mean for normalization
        self._std = std  # Standard deviation for normalization

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self._labels)

    # Get a sample and its label by index
    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert to PIL image
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),  # Randomly resize and crop
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=self._mean, std=self._std)])  # Normalize

        return {
            'image': transform(self._samples[idx]).float(),  # Apply transformations to the image
            'label': torch.from_numpy(self._labels[idx]).float()  # Convert label to tensor
        }

# Function to get train and test data loaders
def get_train_test_loaders(batch_size=32):
    trainset = lengua_senas_dataset('data_2/sign_mnist_train.csv')  # Load training dataset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Create training data loader

    testset = lengua_senas_dataset('data_2/sign_mnist_test.csv')  # Load test dataset
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)  # Create test data loader
    return trainloader, testloader

# Main block to test the data loaders
if __name__ == '__main__':
    loader, _ = get_train_test_loaders(2)  # Get data loaders with batch size of 2
    print(next(iter(loader)))  # Print the first batch of the training loader






