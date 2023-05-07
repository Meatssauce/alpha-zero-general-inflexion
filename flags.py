from enum import Enum

import numpy as np


class PlayerColour(Enum):
    RED = 1, 'R'
    BLUE = -1, 'B'

    def __init__(self, num, token):
        self.num = num
        self.token = token

    @classmethod
    def from_piece(cls, piece: int):
        for player in cls:
            if player.owns(piece):
                return player
        raise IndexError(f'No player owns piece {piece}')

    @property
    def opponent(self):
        for player in self.__class__:
            if player != self:
                return player
        raise IndexError('No opponent')

    def owns(self, piece: int | float | np.ndarray):
        return piece * self.num > 0


class GameOutcome(Enum):
    ONGOING = 0
    DRAW = 1e-4  # draw has some value
    WON = 1
    LOST = -1

    def opposite(self):
        if self == self.WON:
            return self.LOST
        elif self == self.LOST:
            return self.WON
        else:
            return self
