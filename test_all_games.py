""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras.

     [ Games ]      Pytorch      Keras
      -----------   -------      -----
    - Othello        [Yes]       [Yes]
    - TicTacToe                  [Yes]
    - TicTacToe3D                [Yes]
    - Connect4                   [Yes]
    - Gobang                     [Yes]
    - Tafl           [Yes]       [Yes]
    - Rts                        [Yes]
    - DotsAndBoxes               [Yes]
"""

import unittest

import Arena
from MCTS import MCTS

from inflexion.InflexionPlayers import InflexionGame, MCTSPlayer
from inflexion.InflexionPlayers import RandomPlayer
from inflexion.pytorch.NNet import NNetWrapper as InflexionPytorchNNet

import numpy as np

from inflexion.utils import render_board
from utils import *


class TestAllGames(unittest.TestCase):
    @staticmethod
    def execute_game_test(game, neural_net):
        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(neural_net(game), args)
        arena = Arena.Arena(MCTSPlayer(mcts), RandomPlayer(), game, display=render_board)
        print(arena.playGames(2, verbose=True))
   
    def test_inflexion_pytorch(self):
        self.execute_game_test(InflexionGame(7), InflexionPytorchNNet)


if __name__ == '__main__':
    unittest.main()
