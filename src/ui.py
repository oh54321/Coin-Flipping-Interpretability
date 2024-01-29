import pandas as pd
from typing import Optional
from torch import nn
from src.game import CoinGame
from src.agent import CoinAgent
import time

class GameUI:
    
    def __init__(
        self,
        game:CoinGame
    ) -> None:
        self.game = game
    
    def run(
        self,
        model:Optional[nn.Module] = None,
        verbose:bool = True
    ) -> None:
        self.reset()
        done = False
        while not done:
            if verbose: 
                self.print_state()
                time.sleep(0.5)
            action = self._get_action(model)
            if action == "q":
                print("Game Terminated")
                return
            next_state, _, done = self.game.step(action)
            if next_state.num_heads[action]-self.game.curr_state.num_heads[action] == 1 and verbose:
                print("Heads!")
            elif verbose:
                print("Tails!")
        if verbose: 
            self.print_state()
            self._display_done()
    
    def _get_action(
        self,
        model:Optional[CoinAgent] = None
    ) -> int:
        if model is None:
            return self._get_input()
        curr_state = self.game.curr_state
        return model.get_action(curr_state.to_tensor(model.device))
    
    def to_dataframe(
        self
    ) -> pd.DataFrame:
        curr_state = self.game.curr_state
        df = pd.DataFrame.from_dict({
            "Coins":[f"Coin {i}" for i in range(len(curr_state.num_heads))],
            "Heads":curr_state.num_heads,
            "Tails":curr_state.num_tails})
        return df.set_index("Coins")
    
    def print_state(
        self
    ) -> None:
        print("-----------------------")
        print(self.to_dataframe())
        print(f"Number of flips left: {self.game.curr_state.num_flips}") 
        print("-----------------------")
        
    def _get_input(
        self
    ) -> None:
        coin_str = input("What coin do you want to flip? (Enter \"q\" to quit)")
        while not self._valid_coin_input(coin_str):
            coin_str = input("What coin do you want to flip? (Enter \"q\" to quit)")
        return int(coin_str) if coin_str != "q" else coin_str
    
    def reset(
        self
    ) -> None:
        self.game.reset()
    
    def _display_done(
        self
    ) -> None:
        print("No more flips!")
        print(f"Total score: {sum(self.game.curr_state.num_heads)} heads")
        weight_df = pd.DataFrame.from_dict({"index":[0], **{f"Coin {idx}":weight for idx, weight in enumerate(self.game.weights)}})
        print("Weights:")
        print(weight_df)
    
    def _valid_coin_input(
        self,
        inp:str
    ) -> bool:
        if inp == "q":
            return True
        if not inp.isnumeric():
            print("ERROR: you need to input an integer.")
            return False
        if int(inp) >= self.game.num_coins:
            print("ERROR: there are not that many coins.")
            return False
        return True
    
    def score(
        self,
        model,
        num_games:int = 10
    ):
        values = []
        for _ in range(num_games):
            self.run(model, verbose=False)
            values.append(sum(self.game.curr_state.num_heads))
        print(f"Average of {sum(values)/num_games} heads across {num_games} games")