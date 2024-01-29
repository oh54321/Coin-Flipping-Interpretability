import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Sequential
from typing import Tuple, Callable

def _plot_out_1d(
    output, 
    name:str, 
    transform:Callable
) -> None:
    output_np = transform(output.unsqueeze(0).detach().numpy())
    plt.imshow(output_np, cmap='RdBu_r', interpolation='nearest', vmin=-5, vmax=5)
    plt.colorbar()
    plt.title(name)
    plt.show()

def activation_lens(
    model:Sequential, 
    state:torch.Tensor, 
    figsize:Tuple[int]=(4, 4), 
    index = None, 
    transform = lambda x : x
) -> None:
    if index is not None:
        output = model[:index+1](state)
        _plot_out_1d(output, f"Layer {index}", transform = transform)
        return


    output = state
    _plot_out_1d(output, "Input")

    for num, layer in enumerate(model):
        output = layer(output)
        _plot_out_1d(transform(output), f"Layer {num}", transform = transform)