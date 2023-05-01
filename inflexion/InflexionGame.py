from __future__ import print_function
import sys
from collections import defaultdict
from enum import Enum
from itertools import product

from flags import PlayerColour, GameStatus
from inflexion.utils import render_board

sys.path.append('..')
from Game import Game
import numpy as np


class InflexionGame(Game):
    class Move(Enum):
        SPREAD_1 = 0
        SPREAD_2 = 1
        SPREAD_3 = 2
        SPREAD_4 = 3
        SPREAD_5 = 4
        SPREAD_6 = 5
        SPAWN = 6

        @classmethod
        def all_spreads(cls):
            return cls.SPREAD_1, cls.SPREAD_2, cls.SPREAD_3, cls.SPREAD_4, cls.SPREAD_5, cls.SPREAD_6

    def __init__(self, n: int, first_mover: PlayerColour = PlayerColour.RED, curr_player: PlayerColour = None,
                 board: np.ndarray = None, curr_turn: int = 0, max_turns: int = 100, max_power: int = 6):
        super().__init__(n)
        self.max_actions_per_cell = 6 + 1
        self.directions = np.array([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
        self.directions.flags.writable = False

        self.first_mover = first_mover
        self._player = first_mover if curr_player is None else curr_player
        if board is None:
            self.board = np.zeros((n, n), dtype=np.int8)
        else:
            self.board = board
        self.curr_turn = curr_turn
        self.max_turns = max_turns
        self.max_power = max_power

        self.game_status = GameStatus.ONGOING
        self.action_size = self.max_actions_per_cell * self.n ** 2

    @property
    def canonical_board(self):
        return self.board * self.player.num

    @property
    def player(self):
        return self._player

    @player.setter
    def player(self, player: PlayerColour):
        if player == self._player:
            return
        self._player = player
        try:
            self.game_status = self.game_status.opposite()
        except AttributeError:
            raise ValueError("game_status must be of type GameStatus")

    @classmethod
    def from_game(cls, game: 'InflexionGame'):
        try:
            return cls(game.n, first_mover=game.first_mover, curr_player=game.player, board=game.board.copy(),
                       curr_turn=game.curr_turn, max_turns=game.max_turns, max_power=game.max_power)
        except AttributeError:
            raise ValueError("game must be an instance of InflexionGame")

    def reset(self):
        return InflexionGame(self.n, first_mover=self.first_mover, max_turns=self.max_turns, max_power=self.max_power)

    def getNextState(self, action):
        move = self.actionToMove(action)
        new_game = InflexionGame.from_game(self)
        new_game.execute_move(move)
        assert new_game.curr_turn == self.curr_turn + 1
        return new_game, self.player.opposite()

    def getValidMovesMask(self):
        valids = np.zeros((self.n, self.n, self.max_actions_per_cell))
        legal_moves = np.array(self.get_legal_moves(self.player))
        valids[tuple(legal_moves.T)] = 1
        return valids.ravel()

    def moveToAction(self, move: tuple):
        try:
            r, q, move_type = move
        except ValueError:
            raise ValueError("move must be a tuple of length 3")
        return r * self.n * self.max_actions_per_cell + q * self.max_actions_per_cell + move_type

    def actionToMove(self, action: int):
        max_actions_per_row = self.n * self.max_actions_per_cell
        action_idx_in_row = action % max_actions_per_row
        q = action_idx_in_row // self.max_actions_per_cell
        move_type = action_idx_in_row % self.max_actions_per_cell
        r = action // max_actions_per_row
        return r, q, move_type

    def getSymmetries(self, pi: list):
        """Get symmetries of the canonical board and pi"""
        try:
            pi_board = np.reshape(pi, (self.n, self.n, self.max_actions_per_cell))
        except ValueError:
            raise ValueError("pi must be of length self.getActionSize")

        symmetric_boards = []

        for nb_rotation, do_flip in product(range(2), [True, False]):
            new_board = np.rot90(self.canonical_board, 2 * nb_rotation)
            new_pi = np.rot90(pi_board, 2 * nb_rotation)
            if do_flip:
                new_board = np.flipud(np.fliplr(new_board))
                new_pi = np.flipud(np.fliplr(new_pi))
            symmetric_boards.append((new_board, new_pi.ravel().tolist()))
        return symmetric_boards

    def playerCentricBoardBytes(self):
        return self.canonical_board.tobytes()

    def getScore(self, player: PlayerColour):
        return self.count_quantity_diff(player)

    def count_power_diff(self, player: PlayerColour):
        """Count the total power difference for the given player"""
        total = self.board.sum()
        adjusted = player.num * total
        return adjusted

    def count_quantity_diff(self, player: PlayerColour):
        """Count the # pieces difference for the given player"""
        diff = self.board[self.board >= PlayerColour.RED.num].size \
               - self.board[self.board <= PlayerColour.BLUE.num].size
        adjusted = player.num * diff
        return adjusted

    def get_legal_moves(self, player: PlayerColour):
        """Return all legal moves for the player"""
        moves = []
        for r, q in product(range(self.n), range(self.n)):
            new_moves = self.get_moves_for_cell((r, q), player)
            moves += new_moves
        return moves

    def has_legal_moves(self, player: PlayerColour):
        for r, q in product(range(self.n), range(self.n)):
            if self.get_moves_for_cell((r, q), player):
                return True
        return False

    def get_moves_for_cell(self, cell: tuple[int, int], player: PlayerColour):
        try:
            r, q = cell
        except ValueError:
            raise ValueError("cell must be a tuple of length 2")
        if self.board[r, q] == 0:
            return [(r, q, InflexionGame.Move.SPAWN.value)]  # r, q, spawn=0|spread1--6
        if player.owns(self.board[r, q]):
            return [(r, q, InflexionGame.Move(i).value) for i in range(len(self.directions))]
        return []

    def execute_move(self, move):
        def spread_range(origin: tuple[int, int], direction: tuple[int, int]):
            try:
                r, q = origin
                dr, dq = direction
            except ValueError:
                raise ValueError("origin and direction must be tuples of length 2")
            for _ in range(abs(self.board[r, q])):
                r += dr
                r = r % self.n
                q += dq
                q = q % self.n
                yield r, q

        # move encoding: r, q, spawn=0|spread=1--6
        try:
            r, q, move_type = move
        except ValueError:
            raise ValueError("move must be a tuple of length 3")
        move_type = InflexionGame.Move(move_type)

        if move_type == InflexionGame.Move.SPAWN:  # SPAWN
            self.board[r, q] = self.player.num
        elif move_type in InflexionGame.Move.all_spreads():  # SPREAD
            index = (np.array((r, q)) + self.directions) % self.n
            self.board[index] = np.abs(self.board[index]) + 1
            self.board[index] = np.where(self.board[index] > 6, 0, self.board[index]) * self.player.num
            self.board[r, q] = 0
        else:
            raise ValueError("Invalid move")

        self.curr_turn += 1

        # Update game status
        if move_type in InflexionGame.Move.all_spreads() and \
                self.board[self.board >= self.player.opposite().num].size == 0:
            self.game_status = GameStatus.WON
        elif self.curr_turn >= self.max_turns:
            diff = self.count_power_diff(self.player)
            if diff >= 2:
                self.game_status = GameStatus.WON
            elif diff <= -2:
                self.game_status = GameStatus.LOST
            else:
                self.game_status = GameStatus.DRAW
        elif (self.board == 0).all():
            self.game_status = GameStatus.DRAW

    def actionRepr(self, action: int):
        move = self.actionToMove(action)
        try:
            r, q, move_type = move
        except ValueError:
            raise ValueError("move must be a tuple of length 3")
        move_type = InflexionGame.Move(move_type)
        if move_type == InflexionGame.Move.SPAWN:
            return f"Spawn at ({r}, {q})"
        elif move_type in InflexionGame.Move.all_spreads():
            return f"Spread at ({r}, {q}) in direction {move_type.name}"
        else:
            raise ValueError("Invalid move")
