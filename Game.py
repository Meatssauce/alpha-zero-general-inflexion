from flags import PlayerColour


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        pass

    @property
    def board(self):
        raise NotImplementedError

    @property
    def board_size(self):
        raise NotImplementedError

    @property
    def action_size(self):
        raise NotImplementedError

    def getNextState(self, player: PlayerColour, action):
        raise NotImplementedError

    def getValidMoves(self, player: PlayerColour):
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

    def getGameEnded(self, player: PlayerColour):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        raise NotImplementedError

    def getCanonicalForm(self, player: PlayerColour):
        """
        Input:
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        raise NotImplementedError

    def getSymmetries(self, pi):
        """
        Input:
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        raise NotImplementedError

    def stringRepresentation(self):
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
