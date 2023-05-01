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
    def canonicalBoard(self):
        """
        Returns:
            canonicalBoard: a copy of the board in the canonical form of the current player used as nn input
        """
        raise NotImplementedError

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

    def reset(self):
        """
        Returns:
            newGame: a game instance with the same initial parameters in its initial state
        """
        raise NotImplementedError

    def getBoardSize(self):
        """
        Returns:
            boardSize: a tuple of integers (m, n) specifying the board dimensions
        """
        raise NotImplementedError

    def getActionSize(self):
        """
        Returns:
            actionSize: integer number of all possible actions
        """
        raise NotImplementedError

    def getNextState(self, action: int):
        """Take an action, applies it to the current board and returns the next board and player.

        Input:
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        raise NotImplementedError

    def getValidMoves(self):
        """
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        raise NotImplementedError

    def getGameEnded(self):
        """Returns the won/lost/draw/ongoing status of the current game with respect to the current player.

        Returns:
            gameStatus: GameStatus.ONGOING if game has not ended. GameStatus.WON if player won,
                        GameStatus.LOST if player lost, GameStatus.DRAW for draw.
        """
        raise NotImplementedError

    def getSymmetries(self, board, pi):
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

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        raise NotImplementedError

    def getScore(self):
        """
        Returns:
            score: an integer representing the score of the current player
        """
        raise NotImplementedError

    def moveToAction(self, move: tuple):
        """Converts a move tuple to an action integer.

        Input:
            move: a tuple of integers (row, col, ...) representing the move

        Returns:
            action: an integer representing the action
        """
        raise NotImplementedError

    def actionToMove(self, action: int):
        """Converts an action integer to a move tuple.

        Input:
            action: an integer representing the action

        Returns:
            move: a tuple of integers (row, col, ...) representing the move
        """
        raise NotImplementedError
