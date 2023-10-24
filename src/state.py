from typing import List
import torch

class CoinState:
    
    def __init__(
        self,
        num_heads:List[int],
        num_tails:List[int],
        num_flips:int  
    ) -> None:
        assert len(num_heads) == len(num_tails), "Number of heads not equal to tails"
        self.num_heads = num_heads
        self.num_tails = num_tails
        self.num_flips = num_flips
        
    @property
    def num_action(
        self
    ) -> int:
        return len(self.num_heads)
    
    def to_tensor(
        self,
        device = "cuda"
    ) -> torch.Tensor:
        counts = [heads+tails for heads, tails in zip(self.num_heads, self.num_tails)]
        head_fractions = [heads/(heads+tails) if heads+tails > 0 else 0.5 for heads, tails in zip(self.num_heads, self.num_tails)]
        
        return torch.tensor(head_fractions+counts+[self.num_flips], dtype = torch.float32, device = device)
    