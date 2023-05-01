import numpy as np

from Game import Game


class Player:
    def play(self, game):
        assert isinstance(game, Game)


class RandomPlayer(Player):
    def play(self, game):
        super().play(game)
        a = np.random.randint(game.getActionSize())
        valids = game.getValidMoves()
        while valids[a] != 1:
            a = np.random.randint(game.getActionSize())
        return a


class HumanPlayer(Player):
    def play(self, game):
        # display(board)
        valid = game.getValidMoves()
        for i in range(len(valid)):
            r, q, i = game.actionToMove(i)
            print("[", r, q, i, end="] ")
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
    def play(self, game):
        super().play(game)
        valids = game.getValidMoves()
        candidates = []
        for a in range(game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = game.getNextState(a)
            score = game.getScore()
            candidates += [(score, a)]
        candidates.sort(reverse=True)
        return candidates[0][1]


class MCTSPlayer(Player):
    def __init__(self, mcts):
        self.mcts = mcts

    def play(self, game):
        super().play(game)
        return np.argmax(self.mcts.getActionProb(game, temp=0))
