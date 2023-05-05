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
            return cls.SPREAD_R1, cls.SPREAD_R2, cls.SPREAD_Q1, cls.SPREAD_Q2, cls.SPREAD_P1, cls.SPREAD_P2

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
        self.policyShape = self.n, self.n, self.maxActionsPerCell
        # self.movesHistory = []

    def reset(self) -> 'InflexionGame':
        return InflexionGame(self.n, firstMover=self.firstMover, maxTurns=self.maxTurns, maxPower=self.maxPower)

    def getCanonicalBoard(self) -> np.ndarray:
        return self.board * self.player.num

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
        return nextGame, self.player.opponent

    def getValidMoves(self) -> np.ndarray:
        valids = np.zeros(self.policyShape)
        legalMoves = np.array(self.getLegalMoves(self.player))
        valids[tuple(legalMoves.T)] = 1
        return valids.ravel()

    def getGameEnded(self) -> GameStatus:
        return self.gameStatus

    def getSymmetries(self, board: np.ndarray, pi: np.ndarray) -> list[tuple[np.ndarray, list]]:
        assert isinstance(board, np.ndarray) and board.shape == (self.n, self.n)
        assert isinstance(pi, np.ndarray) and pi.size == self.actionSize

        pi = np.reshape(pi, self.policyShape)

        symmetricBoards = self.rotationalSymmetries(board)
        translations = []
        for image in symmetricBoards:
            translations += self.translations(image)
        symmetricBoards += translations

        symmetricPis = self.rotationalSymmetries(pi)
        translations = []
        for image in symmetricPis:
            translations += self.translations(image)
        symmetricPis += translations

        return [(board_, pi_.ravel().tolist()) for board_, pi_ in zip(symmetricBoards, symmetricPis)]

    def translations(self, boardLike: np.ndarray) -> list[np.ndarray]:
        """Returns all translations of the board along the r, q, and s axes.

        Args:
            boardLike: the numpy array of shape (n, n, ...) to be translated

        Returns:
            a list of all translations of the board
        """
        assert isinstance(boardLike, np.ndarray) and boardLike.shape[:2] == (self.n, self.n)
        translatedBoards = []
        for i in range(1, self.n):
            translatedBoards += [
                # translation along r axis
                np.roll(boardLike, i, axis=0),
                # translation along q axis
                np.roll(boardLike, i, axis=1),
                # translation along s axis
                np.roll(np.roll(boardLike, i, axis=1), -i, axis=0)
            ]
        return translatedBoards

    def rotationalSymmetries(self, boardLike: np.ndarray) -> list[np.ndarray]:
        """
        Returns all rotational symmetries of the board.
        Args:
            boardLike: the board to be rotated

        Returns:
            a list of all rotational symmetries of the board

        Hexagonal rotation works like this given r, q, s == 1, 2, 3
            1, 2, 3 => 1, 2, 3  # 0
            1, 2, 3 => -2, 3, 1  # 60
            1, 2, 3 => -3, 1, -2  # 120
            1, 2, 3 => -1, -2, -3  # 180
            1, 2, 3 => 2, -3, -1  # 240
            1, 2, 3 => 3, -1, 2  # 300
        """
        assert isinstance(boardLike, np.ndarray) and boardLike.shape[:2] == (self.n, self.n)
        symmetricBoards = [boardLike.copy()]
        r, q = np.indices((self.n, self.n))
        s = (r + q) % self.n
        orderOfSymmetry = 6  # range(0, 360, 60)
        for i in range(1, orderOfSymmetry):
            r, q, s = (-q) % self.n, s, r
            symmetricBoards.append(boardLike[r, q].copy())
        return symmetricBoards

    def stringRepresentation(self, board):
        return board.tostring()

    def getScore(self) -> int:
        return self.countQuantityDiff(self.player)

    def moveToAction(self, move: tuple | list) -> int:
        r, q, moveType = move
        assert 0 <= r < self.n and 0 <= q < self.n and moveType in InflexionGame.Move
        return r * self.n * self.maxActionsPerCell + q * self.maxActionsPerCell + moveType.num

    def actionToMove(self, action: int) -> tuple[int, int, Move]:
        assert 0 <= action < self.actionSize
        maxActionsPerRow = self.n * self.maxActionsPerCell
        ActionIdxInRow = action % maxActionsPerRow
        q = ActionIdxInRow // self.maxActionsPerCell
        moveType = ActionIdxInRow % self.maxActionsPerCell
        moveType = InflexionGame.Move.fromNum(moveType)
        r = action // maxActionsPerRow
        return r, q, moveType

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
        r, q, moveType = move
        assert 0 <= r < self.n and 0 <= q < self.n and moveType in InflexionGame.Move

        if moveType == InflexionGame.Move.SPAWN:  # SPAWN
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

        self.currTurn += 1
        # self.movesHistory.append((self.player, move))

        # Update game status
        if moveType in InflexionGame.Move.allSpreads() and \
                self.board[self.player.opponent.owns(self.board)].size == 0:
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

    def countPowerDiff(self, player: PlayerColour) -> int:
        """Count the total power difference for the given player"""
        assert player in PlayerColour
        total = self.board.sum()
        adjusted = player.num * total
        return adjusted

    def getLegalMoves(self, player: PlayerColour) -> list[tuple[int, int, int]]:
        """Return all legal moves for the player"""
        assert player in PlayerColour
        moves = []
        for r, q in product(range(self.n), range(self.n)):
            new_moves = self.getMovesForCell((r, q), player)
            moves += new_moves
        return moves

    def getMovesForCell(self, cell: tuple[int, int], player: PlayerColour) -> list[tuple[int, int, int]]:
        assert player in PlayerColour
        assert len(cell) == 2 and all(isinstance(x, int) for x in cell)
        r, q = cell
        if self.board[r, q] == 0:
            return [(r, q, InflexionGame.Move.SPAWN.num)]  # r, q, spawn=0|spread1--6
        if player.owns(self.board[r, q]):
            return [(r, q, move.num) for move in InflexionGame.Move.allSpreads()]
        return []

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
