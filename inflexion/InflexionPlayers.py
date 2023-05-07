import uuid

import numpy as np

from Game import Game
from MCTS import MCTS
from inflexion.InflexionGame import InflexionGame, Move


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
        actions = np.arange(game.max_actions)
        valids = game.valid_actions_mask()
        actions = actions[valids == 1]
        action = np.random.choice(actions)
        return int(action)

    def reset(self):
        return RandomPlayer()


class HumanPlayer(Player):
    def play(self, game: Game) -> int:
        assert isinstance(game, Game)
        # display(board)
        valid = game.valid_actions_mask()
        while True:
            print("Enter move as 3 integer: r q m")
            print("where m is in ", end="")
            print(" ".join([f"{move.name}: {move.num}" for move in Move]))
            input_move = input(">>>")
            r, q, m = [int(x) for x in input_move.split(' ')]
            m = Move.from_num(m)
            try:
                a = game.move_to_action((m, r, q,))
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
        valids = game.valid_actions_mask()
        candidates = []
        for a in range(game.max_actions):
            if valids[a] == 0:
                continue
            nextBoard = game.to_next_state(a)
            nextBoard.player = nextBoard.player.opponent  # switch player back to self
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
