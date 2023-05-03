import uuid

import numpy as np

from Game import Game
from MCTS import MCTS
from inflexion.InflexionGame import InflexionGame


class Player:
    def __init__(self):
        self.id = uuid.uuid4()

    def play(self, game: Game):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.id)


class RandomPlayer(Player):
    def play(self, game: Game):
        assert isinstance(game, Game)
        actions = np.arange(game.getActionSize())
        valids = game.getValidMoves()
        actions = actions[valids == 1]
        action = np.random.choice(actions)
        return int(action)


class HumanPlayer(Player):
    def play(self, game: Game):
        assert isinstance(game, Game)
        # display(board)
        valid = game.getValidMoves()
        while True:
            print("Enter move as 3 integer: r q m")
            print("where m is in ", end="")
            print(" ".join([f"{move.name}: {move.num}" for move in InflexionGame.Move]))
            input_move = input(">>>")
            r, q, m = [int(x) for x in input_move.split(' ')]
            m = InflexionGame.Move.fromNum(m)
            try:
                a = game.moveToAction((r, q, m))
                if valid[a]:
                    break
            except ValueError:
                raise ValueError('Invalid move')
        return a


class GreedyPlayer(Player):
    def play(self, game: Game):
        assert isinstance(game, Game)
        valids = game.getValidMoves()
        candidates = []
        for a in range(game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = game.getNextState(a)
            score = nextBoard.getScore()
            candidates.append((score, a))
        candidates.sort(reverse=True)
        return candidates[0][1]


class MCTSPlayer(Player):
    def __init__(self, mcts: MCTS):
        super().__init__()
        assert isinstance(mcts, MCTS)
        self.mcts = mcts

    def play(self, game: Game):
        assert isinstance(game, Game)
        return int(np.argmax(self.mcts.getActionProb(game, temp=0)))
