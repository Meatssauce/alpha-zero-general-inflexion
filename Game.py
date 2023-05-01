import numpy as np

from flags import PlayerColour


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, n: int):
        self.n = n
        self.board_shape = n, n

    def reset(self):
        raise NotImplementedError

    def getNextState(self, action):
        raise NotImplementedError

    def getValidMovesMask(self):
        """
        Input:
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        raise NotImplementedError

    def moveToAction(self, move: tuple):
        """
        Input:
            move: a move

        Returns:
            action: an action in the form of an integer
        """
        raise NotImplementedError

    def actionToMove(self, action: int):
        """
        Input:
            action: an action in the form of an integer

        Returns:
            move: a move
        """
        raise NotImplementedError

    def getSymmetries(self, board_like: np.ndarray):
        """
        Input:
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        raise NotImplementedError

    def playerCentricBoardBytes(self):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        raise NotImplementedError

    def getScore(self, player: PlayerColour):
        raise NotImplementedError

    def display(self):
        raise NotImplementedError

    def actionRepr(self, action: int):
        raise NotImplementedError
