# Coin-Flipping-Interpretability
**Note:** This project is currently in development and is not intended for external use at this time.

## Overview

Consider the following game: 

*You are given $n$ coins and $k$ coin flips. Each coin has weight $p$, which is obtained by sampling from a normal distribution with mean $0.5$ and standard deviation $\sigma$
 (clipped at $0$ and $1$). How do you flip coins to maximize the number of heads?*

This is inspired by a similar interview question at the quantitative trading firm Jane Street.

The goals of the project are the following:

- Building up classes for training an agent to play this game, with some sort of UI. 
- Training a small 2-layer agent to play this game using DQN. 
- Build up my own smaller intepretability library with the following functionalities:
    - Performing Causal Scrubbing to evaluate circuits.
    - Performing ACDC (automatic circuit discovery code) to figure out circuits for different tasks within the model.
- Use this model to find some explanation for what the agent learns. In particular, I'm using this as some toy experiment to explain exploring/exploiting.

Please note that this is a work in progress.

## Getting Started

To explore some of the project's features and classes, refer to the `main.ipynb` notebook. 

## Further work

I'm putting this project on hold to focus on my work at *SERI MATS*. 

**TO DOs:**

I'm currently running experiments using activation patching methods I learned while working at Redwood research. 

This seems outdated! For example, I know there's been some recent work on attribution patching which allegedly outperforms the above methods.
I also expect I should read more into how this work can draw insights from causal inference; ``removing connections" in activation patching just seems like a causal inference problem and I expect ACDC can be replaced by algorithms used in causal inference. My impression was that there has already been work done in this direction.

So once I resume, I plan to do more literature review and change my approach.
