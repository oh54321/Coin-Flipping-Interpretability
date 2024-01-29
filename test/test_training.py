from src.training import DQNTrainer, ReplayMemory
from src.state import CoinState
import torch
import torch.nn
from torch.optim import Adam
from src.agent import CoinAgent
from src.game import CoinGame
import numpy as np
from torch.nn import functional as F
from src.ui import GameUI
torch.manual_seed(42)

agent = CoinAgent(
    num_coins = 2, 
    num_layers = 1
)
game = CoinGame(
    num_coins = 2,
    num_flips = 2
)
trainer = DQNTrainer(
    max_replay_size=1000,
    batch_size=500,
    gamma=0.99,
    start_epsilon = 0.1,
    end_epsilon = 0.00,
    tau = 0.05,
    num_episodes=5000,
    optimizer=Adam(agent.parameters(), lr=1e-3),
    agent=agent,
    game=game,
    loss_function=F.huber_loss
)
states = [
    CoinState([0, 0], [0, 0], 2),
    CoinState([0, 1], [0, 0], 1),
    CoinState([0, 1], [1, 0], 0)]
actions = [1, 0]
rewards = [1, 0]
dones = [False, True]

def test_lengths():
    trainer.reset()
    agent.train()
    trainer.run_train_loop()
    agent.eval()
    
    memory = ReplayMemory(max_size=3)
    for state, action, next_state, reward, done in zip(states[:-1], actions, states[1:], rewards, dones):
        memory.add(state, action, next_state, reward, done)
    
    sample = memory.sample(3)
    assert len(sample) == 3, f"Sample length different from expected, Expected: 3, Received: {len(sample)}"
    assert len(memory.memory) == 2, f"Memory length different from expected, Expected: 2, Received: {len(memory.memory)}"
    
    for state, action, next_state, reward, done in zip(states[:-1], actions, states[1:], rewards, dones):
        memory.add(state, action, next_state, reward, done)
    print(memory.memory)
    assert len(memory.memory) == 2, f"Memory length different from expected, Expected: 2, Received: {len(memory.memory)}"
    
    memory.reset()
    assert len(memory.memory) == 0, f"Memory length different from expected, Expected: 0, Received: {len(memory.memory)}"
    
    
def test_expected_qs():
    trainer.reset()
    agent.train()
    trainer.run_train_loop()
    agent.eval()
    
    next_state_batch = torch.stack([state.to_tensor() for state in states[1:]])
    reward_tens = torch.tensor(rewards)
    not_dones = torch.tensor([not done for done in dones])

    preds = agent(next_state_batch)
    expected = torch.tensor([1+0.99*max(preds[0]), 0])
    actual = trainer._get_expected_qs(next_state_batch, reward_tens, not_dones)
    assert max((expected-actual).abs()) < 0.001, f"Expected q values different from expected, Expected: {expected}, Received: {actual}"
    
def test_trainer_simple_game():
    trainer.reset()
    agent.train()
    trainer.run_train_loop()
    
    agent.eval()
    gamma = trainer.gamma
    std = trainer.game.standard_dev
    
    state = torch.tensor([
        [0.5, 0.5, 0, 0, 2],
        [0, 0.5, 1, 0, 1],
        [1, 0.5, 1, 0, 1],
        [0.5, 0, 0, 1, 1],
        [0.5, 1, 0, 1, 1]
    ], dtype=torch.float32)
    actual = agent(state)
    expected = torch.tensor([
        [(1+gamma)/2+gamma*std*std, (1+gamma)/2+gamma*std*std],
        [1/2-2*std*std, 1/2],
        [1/2+2*std*std, 1/2],
        [1/2, 1/2-2*std*std],
        [1/2, 1/2+2*std*std]
    ], dtype = torch.float32)
    assert (expected-actual).abs().max(dim = 1)[0].max(dim=0)[0] < 0.001, f"Expected q values different from expected, Expected: {expected}, Received: {actual}"