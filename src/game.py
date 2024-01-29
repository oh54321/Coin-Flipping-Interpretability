from typing import Tuple, Optional
import copy
import numpy as np
from src.state import CoinState

class CoinGame:
    
    def __init__(
        self,
        num_coins:int,
        num_flips:int,
        standard_dev:Optional[float]=0.05
    ) -> None:
        self.num_coins = num_coins
        self.num_flips = num_flips
        self.standard_dev = standard_dev
        self.reset()
    
    def step(
        self, 
        action:int
    ) -> Tuple[CoinState, float, bool]:
        num_heads = copy.deepcopy(self.curr_state.num_heads)
        num_tails = copy.deepcopy(self.curr_state.num_tails)
        num_flips = self.curr_state.num_flips - 1
        if np.random.uniform() < self.weights[action]:
            num_heads[action] += 1
        else:
            num_tails[action] += 1
        reward = self.weights[action]
        new_state = CoinState(num_heads, num_tails, num_flips)
        self.curr_state = new_state
        done = num_flips == 0
        return (new_state, reward, done)
            
    def reset(
        self
    ) -> None:
        self.weights = np.clip(0.5*np.ones(self.num_coins)+np.random.randn(self.num_coins)*self.standard_dev, 0, 1)
        self.curr_state = CoinState([0]*self.num_coins, [0]*self.num_coins, self.num_flips)
