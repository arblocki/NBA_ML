
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class NN(nn.Module):
  def __init__(self, num_inputs=78):
    super().__init__()

    if num_inputs >= 64:
      self.fc1 = nn.Linear(num_inputs, 96)
      self.fc2 = nn.Linear(96, 64)
      self.fc3 = nn.Linear(64, 32)
      self.fc4 = nn.Linear(32, 8)
      self.fc5 = nn.Linear(8, 2)
    elif num_inputs >= 24:
      self.fc1 = nn.Linear(num_inputs, 72)
      self.fc2 = nn.Linear(72, 48)
      self.fc3 = nn.Linear(48, 24)
      self.fc4 = nn.Linear(24, 8)
      self.fc5 = nn.Linear(8, 2)
    else:
      raise ValueError('Low amount of inputs for neural network: ', num_inputs)

    self.init_weights()

  def init_weights(self):
    for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
      FC_in = fc.weight.size(1)
      nn.init.normal_(fc.weight, 0.0, 1 / sqrt(FC_in))
      nn.init.constant_(fc.bias, 0.0)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.fc5(x)

    return x