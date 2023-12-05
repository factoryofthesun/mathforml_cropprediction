import torch
import numpy as np
import torch.nn as nn

### Simple MLP for crop yield regression task
class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d=1, final_activation = None, dropout=True):
        super(MLP, self).__init__()
        """
            input_d: input dimension
            hidden_d: list of dimensions for hidden layers
            output_d: output dimension (defaults to 1 for regression)
            final_activation: maps output to desired range (default: None)
        """
        # Specified dimensions
        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d
        self.dropout = dropout

        self.mlp = [nn.Linear(input_d, hidden_d[0])]
        for di in range(len(hidden_d)-1):
            self.mlp.append(nn.Linear(hidden_d[di], hidden_d[di+1]))
            self.mlp.append(nn.ReLU())
            if dropout:
                self.mlp.append(nn.Dropout())
        self.mlp.append(nn.Linear(hidden_d[-1], output_d))

        if final_activation is not None:
            self.mlp.append(final_activation)

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)
