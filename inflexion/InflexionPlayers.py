import numpy as np

from Game import Game
from flags import PlayerColour
from inflexion.InflexionGame import InflexionGame


class Player:
    def play(self, game):
        raise NotImplementedError


class RandomPlayer(Player):
    def play(self, game):
        valids_mask = game.getValidMovesMask()
        valid_actions = np.argwhere(valids_mask == 1).ravel()
        action = np.random.choice(valid_actions)
        return action


class HumanInflexionPlayer(Player):
    def play(self, game):
        # display(board)
        valid = game.getValidMovesMask()
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i / game.n), int(i % game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 3:
                try:
                    r, q, i = [int(i) for i in input_a]
                    if 0 <= r < game.n and 0 <= q < game.n and 0 <= i < 7:
                        a = game.moveToAction((r, q, i))
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyInflexionPlayer(Player):
    def play(self, game):
        valids = game.getValidMovesMask()
        candidates = []
        for a in range(game.action_size):
            if valids[a] == 0:
                continue
            nextGame, _ = game.getNextState(a)
            score = nextGame.getScore(PlayerColour.RED)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class MCTSPlayer(Player):
    def __init__(self, mcts):
        self.mcts = mcts

    def play(self, game):
        return np.argmax(self.mcts.getActionProb(game, temp=0))
