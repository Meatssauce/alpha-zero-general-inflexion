from enum import Enum


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
        return None

    def opposite(self):
        for player in self.__class__:
            if player != self:
                return player

    def owns(self, piece: int):
        return piece * self.num > 0


class GameStatus(Enum):
    DRAW = 0
    WON = 1
    LOST = -1
    ONGOING = 2

    def opposite(self):
        if self == self.WON:
            return self.LOST
        elif self == self.LOST:
            return self.WON
        else:
            return self

    def score(self):
        if self == self.ONGOING:
            return 0
        return self.value
