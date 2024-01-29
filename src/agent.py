from typing import Iterator, Optional
from torch import nn, Tensor, multinomial
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from src.interp.circuit import CircuitNode

class LinearBlock(nn.Module):
    
    def __init__(
        self,
        size_in:int,
        size_out:int,
        batch_norm:bool = False,
        device:str = "cpu"
    ) -> None:
        super().__init__()
        layer = nn.Linear(size_in, size_out, device = device, bias = False)
        self.model = nn.Sequential(
            layer,
            nn.BatchNorm1d(size_out, device = device)
        ) if batch_norm else layer
    
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


class CircuitAgent:
    
    def __init__(
        self,
        circuit:CircuitNode
    ):
        self.circuit = circuit
        self.device = "cpu"#circuit.device
        
    def get_action(
        self,
        input:Tensor
    ) -> int:
        with torch.no_grad():
            q_vals = self.circuit(input.unsqueeze(0))
            _, best_act = q_vals.max(1)
        return int(best_act)

class CoinAgent(nn.Module):
    
    def __init__(
        self,
        num_coins:int,
        num_layers:int,
        hidden_size:Optional[int] = None,
        device:str = "cpu"
    ) -> None:
        super().__init__()
        if hidden_size is None:
            hidden_size = 2*num_coins+1
        input_layer = LinearBlock(2*num_coins+1, hidden_size, device = device)
        layers = nn.Sequential(*[LinearBlock(hidden_size, hidden_size, device = device) for _ in range(num_layers-1)])
        output_layer = nn.Linear(hidden_size, num_coins, device = device, bias = False)
        self.model = nn.Sequential(input_layer, layers, output_layer)
        self.device = device
    
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