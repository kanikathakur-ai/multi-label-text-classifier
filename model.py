"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the model architecture that you need to implement for HW1.
You should complete the BoWClassifier class by implementing the forward method
and any other necessary components.
"""

import torch
from torch import nn

class BoWClassifier(nn.Module):
    def __init__(self, input_size, num_labels, hidden_size=256, dropout_prob = 0.3):#change drop+hidden
        super().__init__()
        self.num_labels = num_labels
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.reLu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_size, num_labels)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.reLu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
    
def get_best_model(input_size, num_labels):
    return BoWClassifier(input_size=input_size, num_labels=num_labels,hidden_size = 256, dropout_prob = 0.3)#change drop+hidden

def predict(model_output):
    """
    Converts model output to class predictions.
    Args:
        model_output: Output from model.forward(x)
    Returns:
        predictions: Tensor of predicted class labels
    """
    probabilities = torch.sigmoid(model_output)
    predictions = (probabilities > 0.5).float()
    return predictions

