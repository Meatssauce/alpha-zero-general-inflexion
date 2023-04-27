from __future__ import print_function
import sys
from itertools import product

from flags import PlayerColour

sys.path.append('..')
from Game import Game
from .InflexionLogic import Board
import numpy as np


class InflexionGame(Game):
    def __init__(self, n: int, board: np.ndarray = None, curr_turn: int = 0):
        super(Game, self).__init__()
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        if board is not None and not isinstance(board, np.ndarray):
            raise ValueError("board must be a numpy array")
        if not isinstance(curr_turn, int):
            raise ValueError("curr_turn must be an integer")
        self.n = n
        if board is None:
            self._board = np.zeros((n, n), dtype=np.int8)
        else:
            self._board = board
        self.curr_turn = curr_turn

        self.max_turns = 343
        self.directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    @property
    def board(self):
        return self._board

    @property
    def board_size(self):
        return self.n, self.n

    @property
    def action_size(self):
        return (6 + 1) * self.n ** 2

    @classmethod
    def from_game(cls, game: 'InflexionGame'):
        if not isinstance(game, InflexionGame):
            raise ValueError("game must be an instance of InflexionGame")
        new = cls(game.n, np.copy(game.board), game.curr_turn)
        return new

    def invert_board(self):
        self._board = -self._board

    def getNextState(self, player: PlayerColour, action):
        # if player takes action on board, return next (board, player)
        # action must be a valid move
        move = self.actionToMove(action)
        new_game = InflexionGame.from_game(self)
        new_game.execute_move(move, player)
        assert new_game.curr_turn == self.curr_turn + 1
        return new_game, player.opposite()

    def getValidMoves(self, player: PlayerColour):
        # return a fixed size binary vector
        valids = np.zeros(self.action_size).reshape((self.n, self.n, 7))
        legal_moves = self.get_legal_moves(player)
        if len(legal_moves) == 0:
            raise ValueError("No legal moves. This should never occur in Inflexion")
        for r, q, direction_idx in legal_moves:
            valids[r][q][direction_idx] = 1
        return valids.flatten()

    def moveToAction(self, move: tuple):
        # move is a tuple (r, q, direction)
        r, q, direction = move
        return r * self.n * (6 + 1) + q * (6 + 1) + direction

    def actionToMove(self, action: int):
        # action is an integer
        num_actions_per_cell = 6 + 1
        num_actions_per_row = self.n * num_actions_per_cell
        action_idx_in_row = action % num_actions_per_row
        q = action_idx_in_row // num_actions_per_cell
        direction = action_idx_in_row % num_actions_per_cell
        r = action // num_actions_per_row
        return r, q, direction

    def getGameEnded(self, player: PlayerColour):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if self.curr_turn > self.max_turns:
            return 1 if self.countPowerDiff(player) > 0 else -1
        if not self.has_legal_moves(player) or not self.has_legal_moves(player.opposite()):
            return 1
        return 0

    def getCanonicalForm(self, player: PlayerColour):
        # return state if player == 1, else return -state if player == -1
        new_game = InflexionGame.from_game(self)
        new_game.invert_board()
        assert self.curr_turn == new_game.curr_turn
        return new_game

    def getSymmetries(self, pi):
        # mirror, rotational
        if len(pi) != self.action_size:
            raise ValueError("pi must be of length self.getActionSize")
        pi_board = np.reshape(pi, (self.n, self.n, 7))
        symmetric_boards = []

        for nb_rotation, do_flip in product(range(2), [True, False]):
            new_board = np.rot90(self.board, 2 * nb_rotation)
            new_pi = np.rot90(pi_board, 2 * nb_rotation)
            if do_flip:
                new_board = np.flipud(np.fliplr(new_board))
                new_pi = np.flipud(np.fliplr(new_pi))
            symmetric_boards += [(new_board, list(new_pi.ravel()))]
        return symmetric_boards

    def stringRepresentation(self):
        return self.board.tobytes()

    def getScore(self, player: PlayerColour):
        return self.countDiff(player)

    def display(self):
        print("   ", end="")
        for y in range(self.n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(self.n):
            print(y, "|", end="")  # print the row #
            for x in range(self.n):
                piece = self.board[y][x]  # get the piece to print
                print(InflexionGame.get_token(piece), end=" ")
            print("|")

        print("-----------------------")

    @staticmethod
    def get_token(piece):
        if piece != 0:
            return PlayerColour.from_piece(piece).token + abs(piece)
        else:
            return '-'

    def countPowerDiff(self, player: PlayerColour):
        """Count the total power difference for the given colour
        compared to the other.
        (1 for red, -1 for blue, 0 for empty spaces)"""
        total = 0
        for r, q in product(range(self.n), range(self.n)):
            power = self.board[r][q]
            if power == 0:
                continue
            if player.owns(self.board[r][q]):
                total += power
            else:
                total -= power
        return total

    def countDiff(self, player: PlayerColour):
        """Count the # pieces difference for the given colour
        compared to the other.
        (1 for red, -1 for blue, 0 for empty spaces)"""
        count = 0
        for r, q in product(range(self.n), range(self.n)):
            if self.board[r][q] == 0:
                continue
            if player.owns(self.board[r][q]):
                count += 1
            else:
                count -= 1
        return count

    def get_legal_moves(self, player: PlayerColour):
        """Return all legal moves for the given colour.
        (1 for white, -1 for black
        """
        moves = set()
        for r, q in product(range(self.n), range(self.n)):
            if self.board[r][q] == 0 or player.owns(self.board[r][q]):
                new_moves = self.get_moves_for_square((r, q))
                moves.update(new_moves)
        return list(moves)

    def has_legal_moves(self, player: PlayerColour):
        for r, q in product(range(self.n), range(self.n)):
            if self.board[r][q] == 0 or player.owns(self.board[r][q]):
                return True
        return False

    def get_moves_for_square(self, cell):
        """Return all legal moves that use the given cell as a base.
        That is, if the given cell is (3, 4) and it contains a blue piece,
        and (3, 5) and (3, 6) contain red pieces, and (3, 7) is empty, one
        of the returned moves is (3, 7) because everything from there to
        (3, 4) is flipped.
        """
        r, q = cell
        if self.board[r][q] == 0:
            return [(r, q, 0)]  # r, q, spawn=0|spread1--7
        return [(r, q, direction) for direction in range(1, len(self.directions) + 1)]

    def execute_move(self, move, player: PlayerColour):
        """Perform the given move on the board;
        color gives the color pf the piece to play (1=red,-1=blue)
        """
        def spread_range(origin: tuple[int, int], direction: tuple[int, int]):
            r, q = origin
            for _ in range(abs(self.board[r][q])):
                r += direction[0]
                r = r % self.n
                q += direction[1]
                q = q % self.n
                yield r, q

        # move encoding: r, q, spawn=0|spread=1, null=-1|direction=0--6
        r, q, direction_idx = move
        if direction_idx == 0:  # SPAWN
            self.board[r][q] = player.num
        elif 0 < direction_idx < 7:  # SPREAD
            for x_i, y_i in spread_range((r, q), self.directions[direction_idx - 1]):
                self.board[x_i][y_i] = abs(self.board[x_i][y_i]) * player.num + 1
            self.board[r][q] = 0
        else:
            raise ValueError("Invalid move")

        self.curr_turn += 1
