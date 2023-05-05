from __future__ import print_function
import sys
from copy import deepcopy
from enum import Enum
from itertools import product

from flags import PlayerColour, GameStatus

sys.path.append('..')
from Game import Game
import numpy as np


class InflexionGame(Game):
    class Move(Enum):
        SPREAD_R1 = 0, (1, 0)
        SPREAD_R2 = 1, (-1, 0)
        SPREAD_Q1 = 2, (0, 1)
        SPREAD_Q2 = 3, (0, -1)
        SPREAD_P1 = 4, (1, -1)
        SPREAD_P2 = 5, (-1, 1)
        SPAWN = 6, (0, 0)

        def __init__(self, num, direction):
            self.num = num
            self.direction = direction

        @classmethod
        def fromNum(cls, num: int) -> 'InflexionGame.Move':
            for move in cls:
                if move.num == num:
                    return move
            raise IndexError(f"Move number {num} is not valid.")

        @classmethod
        def allSpreads(cls) -> tuple['InflexionGame.Move', ...]:
            return tuple(move for move in cls if move != cls.SPAWN)

    def __init__(self, n: int,
                 firstMover: PlayerColour = PlayerColour.RED,
                 currPlayer: PlayerColour = None,
                 board: np.ndarray = None,
                 currTurn: int = 0,
                 maxTurns: int = 100,
                 maxPower: int = 6):
        super().__init__(firstMover=firstMover)
        assert isinstance(n, int) and n > 0
        assert isinstance(currPlayer, PlayerColour) or currPlayer is None
        assert isinstance(board, np.ndarray) or board is None
        assert board is None or board.shape == (n, n)
        assert isinstance(currTurn, int) and isinstance(maxTurns, int) and isinstance(maxPower, int)

        self.n = n
        if currPlayer is not None:
            self._player = currPlayer
        if board is None:
            self.board = np.zeros((n, n), dtype=np.int8)
        else:
            self.board = board
        self.currTurn = currTurn
        self.maxTurns = maxTurns
        self.maxPower = maxPower

        self.maxActionsPerCell = 6 + 1
        self.actionSize = self.maxActionsPerCell * self.n ** 2
        self.policyShape = self.maxActionsPerCell, self.n, self.n
        self.boardStackShape = 4, self.n, self.n
        self.maxPowerAtSpawn = 48
        # self.movesHistory = []

    def reset(self) -> 'InflexionGame':
        return InflexionGame(self.n, firstMover=self.firstMover, maxTurns=self.maxTurns, maxPower=self.maxPower)

    def toNNetInput(self) -> np.ndarray:
        # return (self.player.num * self.board).reshape(self.boardStackShape)
        playerPieces = (self.player.owns(self.board)).astype(np.int8)
        opponentPieces = (self.player.opponent.owns(self.board)).astype(np.int8)
        totalMoves = np.ones(self.getBoardSize(), dtype=np.int8) * self.currTurn
        canSpawn = np.tile(self.totalPower() <= self.maxPowerAtSpawn, self.getBoardSize()).astype(np.int8)

        return np.stack([playerPieces, opponentPieces, totalMoves, canSpawn], axis=0)

    def isValidAction(self, action):
        move = self.actionToMove(action)
        r, q, moveType = move
        assert 0 <= r < self.n and 0 <= q < self.n and moveType in InflexionGame.Move

        if moveType == InflexionGame.Move.SPAWN:  # SPAWN
            return self.board[r, q] == 0
        elif moveType in InflexionGame.Move.allSpreads():  # SPREAD
            return self.player.owns(self.board[r, q])
        else:
            return False
    
    def nnetInputShape(self):
        return self.boardStackShape
        
    def getBoardSize(self) -> tuple:
        return self.n, self.n

    def getActionSize(self) -> int:
        return self.actionSize

    def getNextState(self, action: int) -> tuple['InflexionGame', PlayerColour]:
        assert isinstance(action, int) and 0 <= action < self.actionSize
        move = self.actionToMove(action)
        nextGame = deepcopy(self)
        nextGame.executeMove(move)
        assert nextGame.currTurn == self.currTurn + 1
        assert nextGame.player != self.player
        return nextGame, self.player.opponent

    def getValidMoves(self) -> np.ndarray:
        valids = np.repeat(self.player.owns(self.board)[np.newaxis, ...], len(InflexionGame.Move), axis=0)
        if self.totalPower() <= self.maxPowerAtSpawn:
            valids[InflexionGame.Move.SPAWN.num, :] = self.board == 0
        else:
            valids[InflexionGame.Move.SPAWN.num, :] = False
        valids = valids.astype(np.int8)
        return valids.ravel()

    def getGameEnded(self) -> GameStatus:
        return self.gameStatus

    def symmetries(self, boardLike: np.ndarray) -> list[np.ndarray]:
        assert (isinstance(boardLike, np.ndarray) and len(boardLike.shape) == len(self.boardStackShape)
                and boardLike.shape[-2:] == self.getBoardSize())

        isomorphicShapes = [boardLike.copy()]
        orderOfRotationalSymmetry = 6

        rotated = [self.rotate(boardLike, i) for i in range(1, orderOfRotationalSymmetry)]
        translated = [self.translate(rotated_, j, axis='r') for rotated_ in rotated for j in range(1, self.n)]
        isomorphicShapes += rotated + translated

        return isomorphicShapes
    
    def randomSymmetry(self, boardLike: np.ndarray) -> np.ndarray:
        assert (isinstance(boardLike, np.ndarray) and len(boardLike.shape) == len(self.boardStackShape)
                and boardLike.shape[-2:] == self.getBoardSize())

        orderOfRotationalSymmetry = 6
        boardLike = self.rotate(boardLike, np.random.randint(0, orderOfRotationalSymmetry))
        boardLike = self.translate(boardLike, np.random.randint(0, self.n), axis=np.random.choice(['r', 'q', 's']))
        return boardLike

    def rotate(self, boardLike: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Returns a board rotated by 60 * k degrees
        Args:
            boardLike: the board to be rotated
            k: the number of 60 degree rotations to perform

        Returns:
            a boardLike array rotated by 60 * k degrees

        Hexagonal rotation works like this given r, q, s == 1, 2, 3
            1, 2, 3 => 1, 2, 3  # 0
            1, 2, 3 => -2, 3, 1  # 60
            1, 2, 3 => -3, 1, -2  # 120
            1, 2, 3 => -1, -2, -3  # 180
            1, 2, 3 => 2, -3, -1  # 240
            1, 2, 3 => 3, -1, 2  # 300

        Sign flips
            1, 1, 1  # 0 translation
            -1, 1, 1  # 1 translation
            -1, 1, -1  # 2 translation
            -1, -1, -1  # 3 translation
            1, -1, -1  # 4 translation
            1, -1, 1  # 5 translation
        """
        assert (isinstance(boardLike, np.ndarray) and len(boardLike.shape) == len(self.boardStackShape)
                and boardLike.shape[-2:] == self.getBoardSize())
        assert isinstance(k, int)

        signFlip = {0: np.array([1, 1, 1]),
                    1: np.array([-1, 1, 1]),
                    2: np.array([-1, 1, -1]),
                    3: np.array([-1, -1, -1]),
                    4: np.array([1, -1, -1]),
                    5: np.array([1, -1, 1])}
        orderOfSymmetry = len(signFlip)  # len(range(0, 360, 60))

        k %= orderOfSymmetry
        r, q = np.indices(self.getBoardSize())
        s = (r + q) % self.n

        r, q, s = np.roll([r, q, s], k, axis=0) * signFlip[k].reshape((-1, 1, 1))

        return boardLike[:, r, q].copy()

    def translate(self, boardLike: np.ndarray, shift: int, axis: str) -> np.ndarray:
        """Returns a board translated by shift cells along axis

        Args:
            boardLike: the numpy array of shape (n, n, ...) to be translated
            shift: the number cells by which to shift the board
            axis: the axis along which to shift the board

        Returns:
            the translated boardLike array
        """
        assert (isinstance(boardLike, np.ndarray) and len(boardLike.shape) == len(self.boardStackShape)
                and boardLike.shape[-2:] == self.getBoardSize())
        assert isinstance(shift, int)
        assert axis in 'rqs'

        match axis:
            case 'r':
                # translation along r axis
                return np.roll(boardLike, shift, axis=1)
            case 'q':
                # translation along q axis
                return np.roll(boardLike, shift, axis=2)
            case 's':
                # translation along s axis
                return np.roll(np.roll(boardLike, shift, axis=2), -shift, axis=1)
        raise ValueError("not supposed to reach here")

    def getScore(self) -> int:
        return self.countQuantityDiff(self.player)

    def moveToAction(self, move: tuple | list) -> int:
        moveType, r, q = move
        assert 0 <= r < self.n and 0 <= q < self.n and moveType in InflexionGame.Move
        return int(np.ravel_multi_index(([moveType.num], [r], [q]), self.policyShape))

    def actionToMove(self, action: int) -> tuple[int, int, Move]:
        assert 0 <= action < self.actionSize
        moveNum, r, q = np.unravel_index(action, self.policyShape)
        moveType = InflexionGame.Move.fromNum(moveNum)
        return moveType, r, q

    def display(self, ansi=False):
        """
        Visualise the Infexion hex board via a multiline ASCII string.
        The layout corresponds to the axial coordinate system as described in the
        game specification document.

        Example:

            # >>> board = {
            # ...     (5, 6): ("r", 2),
            # ...     (1, 0): ("b", 2),
            # ...     (1, 1): ("b", 1),
            # ...     (3, 2): ("b", 1),
            # ...     (1, 3): ("b", 3),
            # ... }
            # >>> print_board(board, ansi=False)

                                    ..
                                ..      ..
                            ..      ..      ..
                        ..      ..      ..      ..
                    ..      ..      ..      ..      ..
                b2      ..      b1      ..      ..      ..
            ..      b1      ..      ..      ..      ..      ..
                ..      ..      ..      ..      ..      r2
                    ..      b3      ..      ..      ..
                        ..      ..      ..      ..
                            ..      ..      ..
                                ..      ..
                                    ..
        """
        board = {}
        for r, q in product(range(self.board.shape[0]), range(self.board.shape[1])):
            piece = self.board[r, q]
            if piece == 0:
                continue
            player = PlayerColour.from_piece(piece)
            power = abs(piece)
            board[(r, q)] = (player.token, power)

        dim = self.n
        output = ""
        for row in range(dim * 2 - 1):
            output += "    " * abs((dim - 1) - row)
            for col in range(dim - abs(row - (dim - 1))):
                # Map row, col to r, q
                r = max((dim - 1) - row, 0) + col
                q = max(row - (dim - 1), 0) + col
                if (r, q) in board:
                    color, power = board[(r, q)]
                    text = f"{color}{power}".center(4)
                    if ansi:
                        output += self.apply_ansi(text, color=color, bold=False)
                    else:
                        output += text
                else:
                    output += " .. "
                output += "    "
            output += "\n"
        print(output)

    def executeMove(self, move: tuple):
        """Execute a move on the board and update the game state."""
        moveType, r, q = move
        assert 0 <= r < self.n and 0 <= q < self.n and moveType in InflexionGame.Move

        if moveType == InflexionGame.Move.SPAWN and self.totalPower() <= self.maxPowerAtSpawn:  # SPAWN
            assert self.board[r, q] == 0
            self.board[r, q] = self.player.num
        elif moveType in InflexionGame.Move.allSpreads():  # SPREAD
            assert self.player.owns(self.board[r, q])
            power = np.abs(self.board[r, q])
            deltas = np.arange(1, power + 1).reshape(power, 1) * np.array(moveType.direction)
            index = (np.array((r, q)) + deltas) % self.n
            index = tuple(index.T)
            self.board[index] = np.abs(self.board[index]) + 1
            self.board[index] = np.where(self.board[index] > 6, 0, self.board[index]) * self.player.num
            self.board[r, q] = 0
        else:
            raise ValueError("Invalid move")

        # Update game status
        if (moveType in InflexionGame.Move.allSpreads() and
                self.board[self.player.opponent.owns(self.board)].size == 0):
            self.gameStatus = GameStatus.WON
        elif self.currTurn >= self.maxTurns:
            diff = self.countPowerDiff(self.player)
            if diff >= 2:
                self.gameStatus = GameStatus.WON
            elif diff <= -2:
                self.gameStatus = GameStatus.LOST
            else:
                self.gameStatus = GameStatus.DRAW
        elif (self.board == 0).all():
            self.gameStatus = GameStatus.DRAW

        self.currTurn += 1
        # self.movesHistory.append((self.player, move))
        self.player = self.player.opponent

    def countPowerDiff(self, player: PlayerColour) -> int:
        """Count the total power difference for the given player"""
        assert player in PlayerColour
        total = self.board.sum()
        adjusted = player.num * total
        return adjusted

    def countQuantityDiff(self, player: PlayerColour) -> int:
        """Count the # pieces difference for the given player"""
        assert player in PlayerColour
        diff = self.board[player.owns(self.board)].size - self.board[player.opponent.owns(self.board)].size
        return diff

    @staticmethod
    def apply_ansi(str: str, bold=True, color=None):
        """
        Wraps a string with ANSI control codes to enable basic terminal-based
        formatting on that string. Note: Not all terminals will be compatible!

        Arguments:

        str -- String to apply ANSI control codes to
        bold -- True if you want the text to be rendered bold
        color -- Colour of the text. Currently only red/"r" and blue/"b" are
            supported, but this can easily be extended if desired...

        """
        bold_code = "\033[1m" if bold else ""
        color_code = ""
        if color == PlayerColour.RED.token:
            color_code = "\033[31m"
        if color == PlayerColour.BLUE.token:
            color_code = "\033[34m"
        return f"{bold_code}{color_code}{str}\033[0m"

    def totalPower(self):
        return np.sum(np.abs(self.board))
