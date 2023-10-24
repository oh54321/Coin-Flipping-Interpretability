# %%
from src.ui import GameUI
from src.training import DQNTrainer
from src.game import CoinGame
from src.agent import CoinAgent
from torch.optim import Adam
import torch
# %%
agent = CoinAgent(
    num_coins = 10, 
    num_layers = 8
)
# agent.load_state_dict(torch.load("model_parameters/trained_agent.pt"))
game = CoinGame(
    num_coins = 10,
    num_flips = 100,
    standard_dev = 0.1
)

ui = GameUI(game)
# %%
trainer = DQNTrainer(
    max_replay_size=10000,
    batch_size=100,
    gamma=0.99,
    start_epsilon = 0.9,
    end_epsilon = 0.05,
    tau = 0.01,
    num_episodes=100,
    optimizer=Adam(agent.parameters(), lr=1e-2),
    agent=agent,
    game=game
)
ui = GameUI(game)
# %%
agent.train()
trainer.run_train_loop()
trainer.show_loss(roll_window = 10)
trainer.show_value(roll_window = 10)
torch.save(agent.state_dict(), "model_parameters/trained_agent.pt")
# %%
agent.eval()
ui.run(model = agent)
# %%
trainer.show_loss(roll_window = 1000)
trainer.show_value(roll_window = 1)
# %%
trainer.show_loss(roll_window = 1000)
trainer.show_value(roll_window = 1000)
# %%
trainer.replay[0]
# %%
