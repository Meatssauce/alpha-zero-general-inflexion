from __future__ import print_function
import sys
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
        def fromNum(cls, num: int):
            for move in cls:
                if move.num == num:
                    return move
            raise IndexError(f"Move number {num} is not valid.")

        @classmethod
        def allSpreads(cls):
            return cls.SPREAD_R1, cls.SPREAD_R2, cls.SPREAD_Q1, cls.SPREAD_Q2, cls.SPREAD_P1, cls.SPREAD_P2

    squareContent = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return InflexionGame.squareContent[piece]

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

    @property
    def canonicalBoard(self):
        return self.board * self.player.num

    @classmethod
    def fromGame(cls, game: 'InflexionGame'):
        assert isinstance(game, InflexionGame)
        return cls(game.n, firstMover=game.firstMover, currPlayer=game.player, board=game.board.copy(),
                   currTurn=game.currTurn, maxTurns=game.maxTurns, maxPower=game.maxPower)

    def reset(self):
        return InflexionGame(self.n, firstMover=self.firstMover, maxTurns=self.maxTurns, maxPower=self.maxPower)

    def getBoardSize(self):
        return self.n, self.n

    def getActionSize(self):
        return self.actionSize

    def getNextState(self, action: int):
        assert isinstance(action, int) and 0 <= action < self.actionSize
        move = self.actionToMove(action)
        nextGame = InflexionGame.fromGame(self)
        nextGame.executeMove(move)
        assert nextGame.currTurn == self.currTurn + 1
        return nextGame, self.player.opponent

    def getValidMoves(self):
        valids = np.zeros(self.policyShape)
        legalMoves = np.array(self.getLegalMoves(self.player))
        valids[tuple(legalMoves.T)] = 1
        return valids.ravel()

    def getGameEnded(self):
        return self.gameStatus

    def getSymmetries(self, board: np.ndarray, pi: np.ndarray):
        assert isinstance(board, np.ndarray) and board.shape == (self.n, self.n)
        assert isinstance(pi, np.ndarray) and pi.size == self.actionSize

        pi = np.reshape(pi, self.policyShape)
        symmetric_boards = []
        for nb_rotation in range(2):
            newBoard = np.rot90(board, 2 * nb_rotation)
            newPi = np.rot90(pi, 2 * nb_rotation)
            symmetric_boards.append((newBoard, newPi.ravel().tolist()))

            for i in range(1, self.n):
                # translation along r axis
                translatedBoard = np.roll(newBoard, i, axis=0)
                translatedPi = np.roll(newPi, i, axis=0)
                symmetric_boards.append((translatedBoard, translatedPi.ravel().tolist()))
                # translation along q axis
                translatedBoard = np.roll(newBoard, i, axis=1)
                translatedPi = np.roll(newPi, i, axis=1)
                symmetric_boards.append((translatedBoard, translatedPi.ravel().tolist()))
                # translation along p axis
                translatedBoard = np.roll(np.roll(newBoard, 1, axis=1), -1, axis=0)
                translatedPi = np.roll(np.roll(newPi, 1, axis=1), -1, axis=0)
                symmetric_boards.append((translatedBoard, translatedPi.ravel().tolist()))

        return symmetric_boards

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.squareContent[square] for row in board for square in row)
        return board_s

    def getScore(self):
        return self.countQuantityDiff(self.player)

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
                print(InflexionGame.squareContent[piece], end=" ")
            print("|")

        print("-----------------------")

    def moveToAction(self, move: tuple | list):
        r, q, moveType = move
        assert 0 <= r < self.n and 0 <= q < self.n and moveType in InflexionGame.Move
        return r * self.n * self.maxActionsPerCell + q * self.maxActionsPerCell + moveType.num

    def actionToMove(self, action: int):
        assert 0 <= action < self.actionSize
        maxActionsPerRow = self.n * self.maxActionsPerCell
        ActionIdxInRow = action % maxActionsPerRow
        q = ActionIdxInRow // self.maxActionsPerCell
        moveType = ActionIdxInRow % self.maxActionsPerCell
        moveType = InflexionGame.Move.fromNum(moveType)
        r = action // maxActionsPerRow
        return r, q, moveType

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

    def countPowerDiff(self, player: PlayerColour):
        """Count the total power difference for the given player"""
        assert player in PlayerColour
        total = self.board.sum()
        adjusted = player.num * total
        return adjusted

    def getLegalMoves(self, player: PlayerColour):
        """Return all legal moves for the player"""
        assert player in PlayerColour
        moves = []
        for r, q in product(range(self.n), range(self.n)):
            new_moves = self.getMovesForCell((r, q), player)
            moves += new_moves
        return moves

    def getMovesForCell(self, cell: tuple[int, int], player: PlayerColour):
        assert player in PlayerColour
        assert len(cell) == 2 and all(isinstance(x, int) for x in cell)
        r, q = cell
        if self.board[r, q] == 0:
            return [(r, q, InflexionGame.Move.SPAWN.num)]  # r, q, spawn=0|spread1--6
        if player.owns(self.board[r, q]):
            return [(r, q, move.num) for move in InflexionGame.Move.allSpreads()]
        return []

    def countQuantityDiff(self, player: PlayerColour):
        """Count the # pieces difference for the given player"""
        assert player in PlayerColour
        diff = self.board[self.board >= PlayerColour.RED.num].size \
               - self.board[self.board <= PlayerColour.BLUE.num].size
        adjusted = player.num * diff
        return adjusted
