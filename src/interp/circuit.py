import torch
import torch.nn
import torch.nn.functional as F
from typing import List, Tuple, Callable, Optional
from abc import abstractmethod
from collections import deque
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import re

from tqdm import tqdm


class CircuitNode:
    
    def __init__(
        self,
        input_nodes:List["CircuitNode"],
        sampler:Callable,
        name:str = "Unnamed Circuit Node",
        has_edge:Optional[List[bool]] = None,
        device:str = "cpu"
    ):
        self.input_nodes = input_nodes
        self.has_edge = [True]*len(input_nodes) if has_edge is None else has_edge
        self.sampler = sampler
        self.name = name
        self.device = device
    
    def __getitem__(self, idx:int):
        return self.input_nodes[idx]
    
    def __repr__(self, level=0):
        indent = " " * 4 * level
        node_repr = f"{indent}{self.name}\n"
        for child in self.input_nodes:
            node_repr += child.__repr__(level + 1)
        return node_repr
    
    def __len__(self):
        return len(self.input_nodes)
    
    def copy(self):
        return CircuitNode(
            input_nodes = [node.copy() for node in self.input_nodes],
            sampler = self.sampler,
            name = self.name,
            device = self.device
        )
    
    @abstractmethod
    def forward(self, input):
        pass
    
    def __call__(self, input):
        return self.forward(input)
    
    def acdc(self, 
        model:torch.nn.Sequential,
        threshold:float,
        loss_fn:Callable,
        num_repeat:int = 5):
            return acdc(self,
            model,
            threshold,
            loss_fn,
            num_repeat)
            
    def visualize(self, figsize = (10, 10)):
        visualize(self, figsize = figsize)
        
    def set_sampler(self, sampler:Callable):
        self.sampler = sampler
        for node in self.input_nodes:
            node.set_sampler(sampler)

class InputNode(CircuitNode):
    
    def __init__(
        self,
        idx:int,
        name:str = "Unnamed Circuit Node",
        device:str = "cpu"
    ):
        super().__init__([], None, name)
        self.idx = idx
        self.name = name
        self.device = device
        
    def forward(
        self,
        input:torch.Tensor
    ):
        return input[self.idx]
    
    def copy(
        self
    ):
        return InputNode(self.idx, self.name, self.device)

class FeedForwardNode(CircuitNode):
    
    def __init__(
        self,
        weights:torch.Tensor,
        input_nodes:List[Union["CircuitNode", InputNode]],
        sampler:Callable,
        name:str = "Unnamed Circuit Node",
        include_active:str = True,
        has_edge:Optional[List[bool]] = None,
        device:str = "cpu"
    ):
        super().__init__(input_nodes, sampler, name, has_edge)
        self.weights = weights.to(device)
        self.include_active = include_active
        self.device = device
        
    def forward(
        self,
        input
    ):
        outputs = []
        for idx, node in enumerate(self.input_nodes):
            if self.has_edge[idx]:
                outputs.append(node.forward(input))
            else:
                outputs.append(node.forward(self.sampler())) 
        output_stacked = torch.stack(outputs)
        pre_activ = self.weights.dot(output_stacked)
        
        return F.relu(pre_activ) if self.include_active else pre_activ
    
    def copy(self):
        return FeedForwardNode(
            weights = self.weights,
            input_nodes = [node.copy() for node in self.input_nodes],
            sampler = self.sampler,
            name = self.name,
            include_active = self.include_active,
            has_edge = deepcopy(self.has_edge),
            device = self.device
        )

class OutputNode(CircuitNode):
    
    def forward(
        self,
        input:torch.Tensor
    ):
        if len(input.shape) == 2:
            return torch.stack([self.forward(inp) for inp in input])
        
        outputs = []
        for idx, node in enumerate(self.input_nodes):
            if self.has_edge[idx]:
                outputs.append(node.forward(input))
            else:
                outputs.append(node.forward(self.sampler()))
                
        return torch.stack(outputs)
    
    def copy(self):
        return OutputNode(
            input_nodes = [node.copy() for node in self.input_nodes],
            sampler = self.sampler,
            name = self.name,
            has_edge = deepcopy(self.has_edge),
            device = self.device
        )

def _create_circuit_helper(
    parameters:List[torch.nn.Parameter],
    idx:int,
    sampler:Callable,
    device:str = "cpu"
):
    """
    Created a circuit from a layer of a standard feedforward NN.

    Args:
        parameters (List[torch.nn.Parameter]): Parameters of FF network.
        idx (int): Index of layer we're creating a circuit for.
        sampler (Callable): Sampler used to sample data according to input distribution.
        device (str, optional): Device for Circuit. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    if idx > 0:
        return [FeedForwardNode(
            weights = weight,
            input_nodes = _create_circuit_helper(parameters, idx-1, sampler),
            sampler = sampler,
            name = f"Layer_{idx}_idx_{idx2}",
            include_active = idx != len(parameters),
            device = device)
            for idx2, weight in enumerate(parameters[idx-1])]
    output_len = parameters[0].shape[1]
    return [InputNode(
        idx2, 
        name = f"Input_{idx2}",
        device = device) for idx2 in range(output_len)]

def _test_edge(
    node:CircuitNode, 
    root_node:CircuitNode, 
    model:torch.nn.Sequential, 
    edge_idx:int,  
    threshold:float,
    loss_fn:Callable,
    num_repeat:int = 5
) -> bool:
    """
    Tests whether an edge should be kept in a circuit by ablating it and measuring the change in some loss function.

    Args:
        node (CircuitNode): Parent node 
        root_node (CircuitNode): Output node of the larger circuit the parent node is a part of.
        model (torch.nn.Sequential): Original model circuit describes.
        edge_idx (int): Index of "child" node
        threshold (float): If difference in loss exceed threshold, we remove the edge.
        loss_fn (Callable): Used to measure difference between circuit and original model giving inputs/outputs.
        num_repeat (int, optional): Number of times to sample and run data on circuit/model to determine whether to keep edge.

    Returns:
        bool: _description_
    """
    node.has_edge[edge_idx] = True
    losses_add, losses_del = [], []
    for _ in range(num_repeat):
        state = root_node.sampler()
        pred, trgt = root_node(state), model(state)
        losses_add.append(loss_fn(pred, trgt))
    node.has_edge[edge_idx] = False
    for _ in range(num_repeat):
        state = root_node.sampler()
        pred, trgt = root_node(state), model(state)
        losses_del.append(loss_fn(pred, trgt))
    diff = abs(sum(losses_add)-sum(losses_del))/num_repeat

    if diff > threshold:
        node.has_edge[edge_idx] = True
    return node.has_edge[edge_idx]

def acdc(
    circuit:CircuitNode,
    model:torch.nn.Sequential,
    threshold:float,
    loss_fn:Callable,
    num_repeat:int = 5
) -> CircuitNode:
    """
    Runs ACDC to detect a sub-circuit of a given circuit.

    Args:
        circuit (CircuitNode): Circuit to 
        model (torch.nn.Sequential): Model circuit is representing.
        threshold (float): We only keep edges so that difference in loss exceeds the threshold when removing them.
        loss_fn (Callable): Loss function.
        num_repeat (int, optional): Number of samples used to calculate loss. Defaults to 5.

    Returns:
        CircuitNode: Sub-circuit returned by ACDC algorithm.
    """
    acdc_circuit = circuit.copy()
    next_layer_nodes = [acdc_circuit]
    layer = 0
    while next_layer_nodes:
        print(f"Layer {layer}")
        layer += 1
        next_layer_temp = []
        for node in tqdm(next_layer_nodes):
            next_layer_temp.extend(
                node[idx]
                for idx in range(len(node))
                if _test_edge(
                    node, acdc_circuit, model, idx, threshold, loss_fn, num_repeat
                )
            )
        next_layer_nodes = next_layer_temp
    return acdc_circuit

def _plot_directed_graph(
    positions:dict, 
    edges:List[Tuple[int]], 
    figsize:Tuple[int] = (10, 10)
) -> None:
    plt.figure(figsize=figsize)
    G = nx.DiGraph()
    G.add_nodes_from(positions.keys())
    G.add_edges_from(edges)
    axs = plt.gca()
    nx.draw_networkx(G, positions, node_size=2000)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize(
    circuit:CircuitNode,
    figsize = (10, 10)
):
    edges = set()
    next_layer_nodes = [circuit]
    depth = 0
    index = 0
    positions = {circuit.name:(0, 0)}
    while next_layer_nodes:
        next_layer_temp = []
        depth += 1
        index = 0
        for node in next_layer_nodes:
            for idx, node2 in enumerate(node.input_nodes):
                if not node.has_edge[idx]:
                    continue
                edges.add((node2.name, node.name))
                if node2.name not in positions:
                    positions[node2.name] = (index, depth)
                    next_layer_temp.append(node2)
                    index += 1
        next_layer_nodes = next_layer_temp
    _plot_directed_graph(positions, edges, figsize = figsize)               
            
            
            
def create_circuit(
    parameters:List[torch.nn.Parameter],
    sampler:Callable,
    device:str = "cpu"
):
    return OutputNode(
        input_nodes=_create_circuit_helper(
            parameters, len(parameters), sampler, device
        ),
        sampler=sampler,
        name="Output",
        device = device
    )
    
    
