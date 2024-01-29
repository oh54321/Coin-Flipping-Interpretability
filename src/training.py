from collections import deque 
import numpy as np
from src.state import CoinState
from typing import Union, List, Tuple
import heapq
import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from src.agent import CoinAgent
from src.game import CoinGame
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class ReplayTuple:
    
    def __init__(
        self,
        state:CoinState,
        action:int,
        next_state:CoinState,
        reward:float,
        done:bool
    ) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done

class ReplayMemory:
    
    def __init__(
        self,
        max_size:int
    ) -> None:
        self.memory = []
        self.max_size = max_size
    
    def add(
        self,
        state:CoinState,
        action:int,
        next_state:CoinState,
        reward:float,
        done:bool
    ) -> None:
        self.memory.append(ReplayTuple(state, action, next_state, reward, done))
        if len(self.memory) > self.max_size:
            self.drop(len(self.memory)//2)
        
    def sample(
        self,
        num_sample:int
    ) -> List[ReplayTuple]:
        assert len(self.memory) > 0, "Memory is empty!"
        return [self.memory[np.random.randint(0, len(self.memory))] for _ in range(num_sample)]
    
    def drop(
        self,
        number:int
    ) -> None:
        self.memory = self.memory[number:]
    
    def reset(
        self
    ) -> None:
        self.memory = []
    
    def __get_item__(
        self,
        index:int
    ) -> ReplayTuple:
        return self.memory[index]

class DQNTrainer:
    
    def __init__(
        self,
        max_replay_size:int,
        batch_size:int,
        gamma:float,
        start_epsilon:float,
        end_epsilon:float,
        tau:float,
        num_episodes:int,
        optimizer:Optimizer,
        agent:CoinAgent,
        game:CoinGame,
        loss_function:nn.Module = nn.MSELoss(),
        device:str = "cpu"
    ) -> None:
        self.replay = ReplayMemory(max_replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.epsilon_increment = (start_epsilon-end_epsilon)/num_episodes
        self.tau = tau
        self.policy = agent
        self.target = copy.deepcopy(agent)
        self.policy.train()
        self.target.eval()
        self.device = device
        self.game = game
        self.optimizer = optimizer
        self.num_episodes = num_episodes
        self.loss_function = loss_function
        self.losses = []
        self.values = []
    
    def run_train_loop(
        self
    ) -> None:
        for _ in tqdm(range(self.num_episodes)):
            self.play_game()
            self.epsilon -= self.epsilon_increment 
        self.epsilon_increment = 0
        
    def update_policy(
        self
    ) -> None:
        state_batch, actions, next_state_batch, rewards, not_dones = self._get_batched_info()
        predicted_qs = self._get_predicted_qs(state_batch, actions)
        expected_qs = self._get_expected_qs(next_state_batch, rewards, not_dones)
        
        # if np.random.uniform() < 0.1:
        #     print(predicted_qs, state_batch[:3], actions, self.policy(next_state_batch), self.target(next_state_batch), expected_qs[:3])
        
        loss = self.loss_function(predicted_qs, expected_qs)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy.parameters(), 10)
        self.optimizer.step()
        self.losses.append(float(loss))
    
    def _get_predicted_qs(
        self,
        state_batch:Tensor,
        actions:Tensor
    ) -> Tensor:
        preds = self.policy.forward(state_batch)
        chosen_qs = preds.gather(1, actions)
        return chosen_qs.squeeze(1)
    
    def _get_expected_qs(
        self,
        next_state_batch:Tensor,
        rewards:Tensor,
        not_dones:Tensor
    ) -> Tensor:
        with torch.no_grad():
            next_q_vec_preds, _ = self.target.forward(next_state_batch).max(dim = 1)
            expected_qs = (self.gamma*next_q_vec_preds)*not_dones.type(torch.float32)+rewards
        return expected_qs
    
    def _get_batched_info(
        self
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        states = self.replay.sample(self.batch_size)
        state_batch = torch.stack([tup.state.to_tensor(device = self.device) for tup in states])
        next_state_batch = torch.stack([tup.next_state.to_tensor(device = self.device) for tup in states])
        rewards = torch.tensor([tup.reward for tup in states], dtype = torch.float32, device = self.device)
        actions = torch.tensor([[tup.action] for tup in states], dtype = torch.int64, device = self.device)
        not_dones = torch.tensor([not tup.done for tup in states], dtype = torch.bool, device = self.device)
        return (state_batch, actions, next_state_batch, rewards, not_dones)
    
    def reset(
        self
    ) -> None:
        self.replay.reset()
        self.game.reset()
        self.losses = []
    
    def update_target(
        self
    ) -> None:
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target.load_state_dict(target_net_state_dict)
    
    def play_game(
        self
    ) -> None:
        self.game.reset()
        done = False
        value = 0
        while not done:
            self.policy.eval()
            curr_state = self.game.curr_state
            action = self.policy.get_action(curr_state.to_tensor(device = self.device))
            if np.random.uniform() < self.epsilon:
                action = np.random.randint(0, self.game.num_coins)
            next_state, reward, done = self.game.step(action)
            self.replay.add(curr_state, action, next_state, reward, done)
            value += reward
            
            self.policy.train()
            self.update_policy()
            self.update_target()
        self.values.append(value)
        
    def show_loss(
        self, 
        roll_window=10
    ) -> None:
        rolling_avg = np.convolve(self.losses, np.ones(roll_window)/roll_window, mode='valid')
        x = np.arange(roll_window-1, len(self.losses))
        plt.figure(figsize=(10, 6))
        plt.plot(x, rolling_avg, label=f'Rolling Average ({roll_window}-element)')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss Over Training with Rolling Average')
        plt.legend()
        plt.show()
    
    def show_value(
        self, 
        roll_window=10
    ) -> None:
        rolling_avg = np.convolve(self.values, np.ones(roll_window)/roll_window, mode='valid')
        x = np.arange(roll_window-1, len(self.values))
        plt.figure(figsize=(10, 6))
        plt.plot(x, rolling_avg, label=f'Rolling Average ({roll_window}-element)')
        plt.xlabel('Training Step')
        plt.ylabel('Total Value')
        plt.title('Total Value Over Training with Rolling Average')
        plt.legend()
        plt.show()