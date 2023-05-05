import uuid

import numpy as np

from Game import Game
from MCTS import MCTS
from inflexion.InflexionGame import InflexionGame


class Player:
    def __init__(self):
        self.id = uuid.uuid4()

    def play(self, game: Game) -> int:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.id)

    def reset(self):
        raise NotImplementedError


class RandomPlayer(Player):
    def play(self, game: Game) -> int:
        assert isinstance(game, Game)
        actions = np.arange(game.getActionSize())
        valids = game.getValidMoves()
        actions = actions[valids == 1]
        action = np.random.choice(actions)
        return int(action)

    def reset(self):
        return RandomPlayer()


class HumanPlayer(Player):
    def play(self, game: Game) -> int:
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

    def reset(self):
        return HumanPlayer()


class GreedyPlayer(Player):
    def play(self, game: Game) -> int:
        assert isinstance(game, Game)
        valids = game.getValidMoves()
        candidates = []
        for a in range(game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, opponent = game.getNextState(a)
            nextBoard.player = opponent.opponent  # switch player back to self
            score = nextBoard.score()
            candidates.append((score, a))
        candidates.sort(reverse=True)
        return candidates[0][1]

    def reset(self):
        return GreedyPlayer()


class MCTSPlayer(Player):
    def __init__(self, mcts: MCTS):
        super().__init__()
        assert isinstance(mcts, MCTS)
        self.mcts = mcts

    def play(self, game: Game):
        assert isinstance(game, Game)
        return int(np.argmax(self.mcts.getActionProb(game, temp=0)))

    def reset(self):
        return MCTSPlayer(self.mcts.reset())
