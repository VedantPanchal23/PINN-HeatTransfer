import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, layers):
        super(FCN, self).__init__()
        self.activation = nn.Tanh()

        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(self.activation)

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)
