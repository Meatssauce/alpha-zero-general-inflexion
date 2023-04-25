from __future__ import print_function
import sys
from itertools import product

sys.path.append('..')
from Game import Game
from .InflexionLogic import Board
import numpy as np


class InflexionGame(Game):
    def __init__(self, n):
        super(Game, self).__init__()
        self.n = n

    @staticmethod
    def get_token(piece):
        if piece < 0:
            return 'B' + abs(piece)
        if piece > 0:
            return 'R' + abs(piece)
        else:
            return '-'

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return (6 + 1) * self.n ** 2

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board, player)
        # action must be a valid move
        num_actions_per_cell = 6 + 1
        num_actions_per_row = self.n * num_actions_per_cell
        action_idx_in_row = action % num_actions_per_row
        q = action_idx_in_row // num_actions_per_cell
        direction = action_idx_in_row % num_actions_per_cell
        r = action // num_actions_per_row
        move = (r, q, direction)

        b = Board.from_board(board)
        b.execute_move(move, player)
        return board.pieces, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board.from_board(board)
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            raise ValueError("No legal moves. This should never occur in Inflexion")
        for r, q, direction_idx in legal_moves:
            valids[self.n * r + q + direction_idx] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board.from_board(board)
        if not b.has_legal_moves(player) or not b.has_legal_moves(-player):
            return 1
        elif b.get_curr_turn() > b.get_max_turns():
            return 1 if b.countPowerDiff(player) > 0 else -1
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player == 1, else return -state if player == -1
        if player != 1 and player != -1:
            raise ValueError("player must be 1 or -1")
        return player * board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        if len(pi) != self.getActionSize():
            raise ValueError("pi must be of length self.getActionSize")
        pi_board = np.reshape(pi, (self.n, self.n, 7))
        symmetric_boards = []

        for nb_rotation, do_flip in product(range(2), [True, False]):
            new_board = np.rot90(board, 2 * nb_rotation)
            new_pi = np.rot90(pi_board, 2 * nb_rotation)
            if do_flip:
                new_board = np.flipud(np.fliplr(new_board))
                new_pi = np.flipud(np.fliplr(new_pi))
            symmetric_boards += [(new_board, list(new_pi.ravel()))]
        return symmetric_boards

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(InflexionGame.get_token(cell) for row in board for cell in row)
        return board_s

    def getScore(self, board, player):
        b = Board.from_board(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                print(InflexionGame.get_token(piece), end=" ")
            print("|")

        print("-----------------------")
