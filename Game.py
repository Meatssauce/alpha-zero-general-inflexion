import numpy as np

from flags import PlayerColour, GameStatus


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    """
    def __init__(self, firstMover: PlayerColour):
        assert isinstance(firstMover, PlayerColour)
        self.firstMover = firstMover
        self._player = firstMover
        self.gameStatus = GameStatus.ONGOING

    @property
    def player(self):
        return self._player

    @player.setter
    def player(self, player: PlayerColour):
        """Set the current player.

        Setting the opponent as current player automatically changes the game status.

        Args:
            player: the player to set as current player
        """
        if player == self._player:
            return
        assert isinstance(player, PlayerColour)
        self._player = player
        self.gameStatus = self.gameStatus.opposite()

    def reset(self) -> 'Game':
        """
        Returns:
            newGame: a game instance with the same initial parameters in its initial state
        """
        raise NotImplementedError

    def getBoardSize(self) -> tuple:
        """
        Returns:
            boardSize: a tuple of integers (m, n) specifying the board dimensions
        """
        raise NotImplementedError

    def getActionSize(self) -> int:
        """
        Returns:
            actionSize: integer number of all possible actions
        """
        raise NotImplementedError

    def getNextState(self, action: int) -> tuple['Game', PlayerColour]:
        """Take an action, applies it to the current board and returns the next board and player.

        Following applying action, GameStatus and current turn are updated, and the current player is switched to the
        opponent.

        Input:
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        raise NotImplementedError

    def getValidMoves(self) -> np.ndarray:
        """
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        raise NotImplementedError

    def getGameEnded(self) -> GameStatus:
        """Returns the won/lost/draw/ongoing status of the current game with respect to the current player.

        Returns:
            gameStatus: GameStatus.ONGOING if game has not ended. GameStatus.WON if player won,
                        GameStatus.LOST if player lost, GameStatus.DRAW for draw.
        """
        raise NotImplementedError

    def symmetries(self, boardLike: np.ndarray) -> list[tuple[np.ndarray, list]]:
        """Get all symmetric forms of the board and pi vector.

        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        raise NotImplementedError

    def randomSymmetry(self, boardLike: np.ndarray) -> np.ndarray:
        """Get a random symmetric form of the board.

        Call this function prior to inference in mcts to average out bias

        Input:
            boardLike: one or more planes with the same last 2 dimensions as the board

        Returns:
            symmBoard: a symmetrical form of the board
        """
        raise NotImplementedError

    def getScore(self) -> int:
        """
        Returns:
            score: an integer representing the score of the current player
        """
        raise NotImplementedError

    def moveToAction(self, move: tuple | list) -> int:
        """Converts a move tuple to an action integer.

        Input:
            move: a tuple or list (row, col, ...) representing the move

        Returns:
            action: an integer representing the action
        """
        raise NotImplementedError

    def actionToMove(self, action: int) -> tuple[int, int, int]:
        """Converts an action integer to a move tuple.

        Input:
            action: an integer representing the action

        Returns:
            move: a tuple of integers (row, col, ...) representing the move
        """
        raise NotImplementedError

    def display(self):
        """Displays the current board."""
        raise NotImplementedError

    def toNNetInput(self):
        """Converts the current board to the input of the neural network.

        Returns:
            boardStack: a stack of (M + L) x N1 x N2 board-like planes representing various features of the board
                        where N1, N2 are the board's dimensions, M is the number of features and L is the number of
                        constant valued planes representing the state of hidden variables and special rules
        """
        raise NotImplementedError
