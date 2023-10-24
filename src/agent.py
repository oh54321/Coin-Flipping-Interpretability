from typing import Iterator
from torch import nn, Tensor, multinomial
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

class LinearBlock(nn.Module):
    
    def __init__(
        self,
        size_in:int,
        size_out:int,
        device:str = "cuda"
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size_in, size_out, device = device),
            nn.BatchNorm1d(size_out, device = device)
        )
    
    def forward(
        self,
        input:Tensor
    ) -> Tensor:
        return F.relu(self.model(input))

    def parameters(
        self, 
        recurse: bool = True
    ) -> Iterator[Parameter]:
        return self.model.parameters(recurse)
    
    def train(
        self,
        *args,
        **kwargs
    ) -> None:
        self.model.train(*args, **kwargs)

    def eval(
        self,
        *args,
        **kwargs
    ) -> None:
        self.model.eval(*args, **kwargs)
    
class CoinAgent(nn.Module):
    
    def __init__(
        self,
        num_coins:int,
        num_layers:int,
        device:str = "cuda"
    ) -> None:
        super().__init__()
        layers = nn.Sequential(*[LinearBlock(2*num_coins+1, 2*num_coins+1, device = device) for _ in range(num_layers)])
        output_layer = nn.Linear(2*num_coins+1, num_coins, device = device)
        self.model = nn.Sequential(layers, output_layer)
    
    def forward(
        self,
        input:Tensor
    ) -> Tensor:
        return self.model(input)
    
    def get_action(
        self,
        input:Tensor
    ) -> int:
        with torch.no_grad():
            q_vals = self(input.unsqueeze(0))
            _, best_act = q_vals.max(1)
        return int(best_act)
    
    def parameters(
        self, 
        recurse: bool = True
    ) -> Iterator[Parameter]:
        return self.model.parameters(recurse)
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()