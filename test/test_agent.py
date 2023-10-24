import torch
from torch import nn
from src.agent import CoinAgent, LinearBlock
torch.manual_seed(42)

def check_module(
    model:nn.Module,
    input_dim:int,
    output_dim:int,
    batch_size:int,
    num_epoch:int,
    loss_func:nn.Module = nn.MSELoss()
) -> None:
    model.train()
    expected_single = torch.randn(output_dim)
    expected = torch.stack([expected_single]*batch_size)
    optim = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.MSELoss()
    
    for _ in range(num_epoch):
        optim.zero_grad()
        loss = loss_func(expected, model(torch.randn((batch_size, input_dim))))
        loss.backward()
        optim.step()
    
    model.eval()
    test_exp = expected_single.unsqueeze(0)
    test_pred = model(torch.randn((1, input_dim)))
    assert loss_func(test_exp, test_pred) < 0.001, "Moduel didn't learn expected output!"

def test_linear_block_trains() -> None:
    input_dim = 3
    model = torch.nn.Sequential(
        LinearBlock(input_dim, input_dim),
        torch.nn.Linear(input_dim, input_dim)
    )
    check_module(
        model = model,
        input_dim = input_dim,
        output_dim = input_dim,
        batch_size = 100,
        num_epoch = 5000
    )

def test_coin_agent_trains() -> None:
    num_coin = 3
    model = CoinAgent(num_coins = 3, num_layers = 2)
    check_module(
        model = model,
        input_dim = 2*num_coin+1,
        output_dim = num_coin,
        batch_size = 100,
        num_epoch = 5000
    )

def test_coin_agent_action() -> None:
    num_coin = 3
    model = CoinAgent(num_coins = 3, num_layers = 2)
    model.eval()
    input = torch.randn(2*num_coin+1)
    output = model(input.unsqueeze(0)).squeeze(0)
    _, amax = output.max(0)

    assert int(amax) == model.get_action(input), "Not correct best action!"