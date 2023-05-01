import Arena
from MCTS import MCTS
from inflexion.InflexionGame import InflexionGame
from inflexion.InflexionPlayers import *
from inflexion.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

game = InflexionGame(7, maxTurns=343, maxPower=6)

# all players
rp = RandomPlayer()
gp = GreedyPlayer()
hp = HumanPlayer()

# nnet players
n1 = NNet(game)
n1.load_checkpoint('./pretrained_models/inflexion/pytorch/', '7x7_100checkpoints_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(n1, args1)
n1p = MCTSPlayer(mcts1)

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(game)
    n2.load_checkpoint('./pretrained_models/inflexion/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(n2, args2)
    n2p = MCTSPlayer(mcts2)

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, game, display=InflexionGame.display)

print(arena.playGames(2, verbose=True))
