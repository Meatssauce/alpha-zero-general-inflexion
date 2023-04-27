"""
Author: Zhehong Zhang
Date: April 25, 2023.
Board class.
Board data:
  positives=red, negatives=blue, 0=empty
  first dim is row , 2nd is column:
     pieces[1][7] is the cell in row 2,
     at the opposite end of the board in column 8.
Cells are stored and manipulated as (r,q) tuples.
r is the row, q is the column.
"""
from itertools import product
from typing import Iterable

import numpy as np


class Board:
    # list of all 8 directions on the board, as (r,q) offsets
    __directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    __MAX_TURNS = 343

    def __init__(self, n):
        """Set up initial board configuration."""
        self.n = n
        # Create the empty board array.
        self.pieces = [[0] * self.n for _ in range(self.n)]
        self.curr_turn = 0

    @classmethod
    def from_board(cls, board):
        new_board = Board(board.n)
        new_board.pieces = np.copy(board.pieces)
        new_board.curr_turn = board.curr_turn
        return new_board

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_curr_turn(self):
        return self.curr_turn

    def get_max_turns(self):
        return self.__MAX_TURNS

    def countPowerDiff(self, color):
        """Counts the total power difference of the given color
        (1 for red, -1 for blue, 0 for empty spaces)"""
        total = 0
        for r, q in product(range(self.n), range(self.n)):
            power = self[r][q]
            if power == 0:
                continue
            if self._same_colour(color, r, q):
                total += power
            else:
                total -= power
        return total

    def countDiff(self, color):
        """Counts the # pieces difference of the given color
        (1 for red, -1 for blue, 0 for empty spaces)"""
        count = 0
        for r, q in product(range(self.n), range(self.n)):
            if self[r][q] == 0:
                continue
            if self._same_colour(color, r, q):
                count += 1
            else:
                count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for r, q in product(range(self.n), range(self.n)):
            if self[r][q] == 0 or self._same_colour(color, r, q):
                new_moves = self.get_moves_for_square((r, q))
                moves.update(new_moves)
        return list(moves)

    def has_legal_moves(self, color):
        for r, q in product(range(self.n), range(self.n)):
            if self[r][q] == 0 or self._same_colour(color, r, q):
                return True
        return False

    def _same_colour(self, color, r, q):
        if color != 1 and color != -1:
            raise ValueError("Invalid color")
        return (self[r][q] > 0 and color > 0) or (self[r][q] < 0 and color < 0)

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given cell as a base.
        That is, if the given cell is (3, 4) and it contains a blue piece,
        and (3, 5) and (3, 6) contain red pieces, and (3, 7) is empty, one
        of the returned moves is (3, 7) because everything from there to
        (3, 4) is flipped.
        """
        r, q = square
        if self[r][q] == 0:
            return [(r, q, 0)]  # r, q, spawn=0|spread1--7
        return [(r, q, direction) for direction in range(1, len(self.__directions) + 1)]

    def execute_move(self, move, color):
        """Perform the given move on the board;
        color gives the color pf the piece to play (1=red,-1=blue)
        """
        if color != 1 and color != -1:
            raise ValueError("Invalid color")

        # move encoding: r, q, spawn=0|spread=1, null=-1|direction=0--6
        r, q, direction_idx = move
        if direction_idx == 0:  # SPAWN
            self[r][q] = color
        elif direction_idx == 1:  # SPREAD
            for x_i, y_i in self._spread_range((r, q), self.__directions[direction_idx - 1]):
                self[x_i][y_i] = abs(self[x_i][y_i]) * color + 1
            self[r][q] = 0
        else:
            raise ValueError("Invalid move")

        self.curr_turn += 1

    def _spread_range(self, origin: tuple[int, int], direction: tuple[int, int]):
        """ Generator expression for incrementing moves """
        r, q = origin
        for _ in range(abs(self[r][q])):
            r += direction[0] % self.n
            q += direction[1] % self.n
            yield r, q
