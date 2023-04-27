import numpy as np

from Game import Game
from flags import PlayerColour
from inflexion.InflexionGame import InflexionGame


class Player:
    def __init__(self, game: Game):
        if not isinstance(game, Game):
            raise ValueError("game must be an instance of Game")
        self.game = game

    def play(self, game: Game):
        raise NotImplementedError


class RandomPlayer(Player):
    def play(self, game: Game):
        a = np.random.randint(game.action_size)
        valids = game.getValidMoves(PlayerColour.RED)
        while valids[a] != 1:
            a = np.random.randint(game.action_size)
        return a


class HumanInflexionPlayer(Player):
    def play(self, game: Game):
        # display(board)
        valid = game.getValidMoves(PlayerColour.RED)
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
    def play(self, game: Game):
        valids = game.getValidMoves(PlayerColour.RED)
        candidates = []
        for a in range(game.action_size):
            if valids[a] == 0:
                continue
            nextGame, _ = game.getNextState(PlayerColour.RED, a)
            score = nextGame.getScore(PlayerColour.RED)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
