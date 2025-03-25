from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

import onnx
import onnxruntime as ort

from dataset import get_train_test_loaders
from neural_network import Net

def evaluate(outputs: Variable, labels: Variable) -> float:
    """Evaluate neural network outputs against non-one-hotted labels."""
    Y = labels.numpy()  # Convert labels to numpy array
    Yhat = np.argmax(outputs, axis=1)  # Get the index of the probability classifications
    return float(np.sum(Yhat == Y))  # Calculate the number of correct predictions

def batch_evaluate(
        net: Net,
        dataloader: torch.utils.data.DataLoader) -> float:
    """Evaluate neural network in batches, if dataset is too large."""
    score = n = 0.0  # Initialize score and sample count
    for batch in dataloader:
        n += len(batch['image'])  # Update sample count
        outputs = net(batch['image'])  # Get model outputs
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()  # Convert outputs to numpy array if they are tensors
        score += evaluate(outputs, batch['label'][:, 0])  # Update score with batch evaluation
    return score / n  # Return average score

def validate():
    trainloader, testloader = get_train_test_loaders()  # Load training and testing data
    net = Net().float().eval()  # Initialize the network and set it to evaluation mode

    pretrained_model = torch.load("checkpoint.pth")  # Load the pretrained model
    net.load_state_dict(pretrained_model)  # Load the model state into the network

    print('=' * 10, 'PyTorch', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.  # Evaluate training accuracy
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.  # Evaluate validation accuracy
    print('Validation accuracy: %.1f' % test_acc)

    trainloader, testloader = get_train_test_loaders(1)  # Reload data with batch size 1

    # Export to ONNX
    fname = "signlanguage.onnx"
    dummy = torch.randn(1, 1, 28, 28)  # Create a dummy input tensor
    torch.onnx.export(net, dummy, fname, input_names=['input'])  # Export the model to ONNX format

    # Check exported model
    model = onnx.load(fname)  # Load the ONNX model
    onnx.checker.check_model(model)  # Check if the model is well-formed

    # Create runnable session with exported model
    ort_session = ort.InferenceSession(fname)  # Create an ONNX runtime session
    net = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]  # Define a lambda function to run the model

    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.  # Evaluate training accuracy with ONNX model
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.  # Evaluate validation accuracy with ONNX model
    print('Validation accuracy: %.1f' % test_acc)

if __name__ == '__main__':
    validate()  # Run the validate function if the script is executed directly