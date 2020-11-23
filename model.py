import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_node):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2,num_node)
        self.fc2 = nn.Linear(num_node, num_node)
        self.fc3 = nn.Linear(num_node, num_node)
        self.fc3 = nn.Linear(num_node, num_node)
        self.fc4 = nn.Linear(num_node, num_node)
        self.fc5 = nn.Linear(num_node, num_node)
        self.fc6 = nn.Linear(num_node, num_node)
        self.fc7 = nn.Linear(num_node, num_node)
        self.fc8 = nn.Linear(num_node, num_node)
        self.fc9 = nn.Linear(num_node, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        x = self.fc9(x)
        return x