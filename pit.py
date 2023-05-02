from multiprocessing import freeze_support

import Arena
from Game import Game
from MCTS import MCTS
from inflexion.InflexionGame import InflexionGame
from inflexion.InflexionPlayers import RandomPlayer, HumanPlayer, GreedyPlayer, MCTSPlayer
from inflexion.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


def getPlayer(kind: str, game: Game):
    match kind:  # random, greedy, mcts, human
        case "human":
            player = HumanPlayer()
        case "random":
            player = RandomPlayer()
        case "greedy":
            player = GreedyPlayer()
        case "mcts":
            nn = NNet(game)
            nn.load_checkpoint('./dev/models/inflexion/7x343x6/', 'best.pth.tar')
            args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
            mcts = MCTS(nn, args)
            player = MCTSPlayer(mcts)
        case _:
            raise ValueError(f"Unknown CPU player: {kind}")
    return player


def main():
    game = InflexionGame(7, maxTurns=343, maxPower=6)
    player1 = getPlayer("random", game)
    player2 = getPlayer("greedy", game)
    arena = Arena.Arena(player1, player2, game)

    print(arena.playGames(2, verbose=True))


if __name__ == "__main__":
    # freeze_support()
    main()
