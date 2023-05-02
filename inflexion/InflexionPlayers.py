import numpy as np

from Game import Game
from MCTS import MCTS


class Player:
    def play(self, game: Game):
        raise NotImplementedError


class RandomPlayer(Player):
    def play(self, game: Game):
        assert isinstance(game, Game)
        actions = np.arange(game.getActionSize())
        valids = game.getValidMoves()
        actions = actions[valids == 1]
        action = np.random.choice(actions)
        return action


class HumanPlayer(Player):
    def play(self, game: Game):
        assert isinstance(game, Game)
        # display(board)
        valid = game.getValidMoves()
        for i in range(len(valid)):
            r, q, moveType = game.actionToMove(i)
            print("[", r, q, moveType.num, end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            try:
                a = game.moveToAction(input_a)
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
            score = game.getScore()
            candidates.append((score, a))
        candidates.sort(reverse=True)
        return candidates[0][1]


class MCTSPlayer(Player):
    def __init__(self, mcts: MCTS):
        assert isinstance(mcts, MCTS)
        self.mcts = mcts

    def play(self, game: Game):
        assert isinstance(game, Game)
        return np.argmax(self.mcts.getActionProb(game, temp=0))
