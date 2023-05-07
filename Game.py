from typing import Any

import numpy as np

from flags import PlayerColour, GameOutcome


class Game:
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    """
    def __init__(self, board_shape: tuple, policy_shape: tuple, first_mover: PlayerColour):
        assert (isinstance(board_shape, tuple) and len(board_shape) == 2 and
                all(isinstance(i, int) for i in board_shape))
        assert (isinstance(policy_shape, tuple) and len(policy_shape) == 3 and
                all(isinstance(i, int) for i in policy_shape))
        assert isinstance(first_mover, PlayerColour)

        self._board_shape = board_shape
        self._policy_shape = policy_shape
        self._firstMover = first_mover
        self._player = first_mover
        self._outcome = GameOutcome.ONGOING

    @property
    def board_shape(self) -> tuple[int, int]:
        """
        Returns:
            board_shape: a tuple of integers (m, n) specifying the board dimensions
        """
        return self._board_shape

    @property
    def policy_shape(self) -> tuple[int, int, int]:
        """
        Returns:
            policy_shape: a tuple of integers (k, m, n) specifying the policy dimensions
                          k represents the number of possible actions, m and n represent
                          the board dimensions
        """
        return self._policy_shape

    @property
    def player(self) -> PlayerColour:
        return self._player

    @player.setter
    def player(self, player: PlayerColour):
        """Set the current player.

        Setting the opponent as current player automatically changes the game status.

        Args:
            player: the player to set as current player
        """
        assert isinstance(player, PlayerColour)
        if player == self._player:
            return
        self._player = player
        self._outcome = self._outcome.opposite()

    @property
    def outcome(self) -> GameOutcome:
        """Returns the won/lost/draw/ongoing status of the current game with respect to the current player.

        Returns:
            gameStatus: GameOutcome.ONGOING if game has not ended. GameOutcome.WON if player won,
                        GameOutcome.LOST if player lost, GameOutcome.DRAW for draw.
        """
        return self._outcome

    @property
    def max_actions(self) -> int:
        """
        Returns:
            max_actions: the number of all possible actions in the game
        """
        return int(np.prod(self._policy_shape))

    def restarted(self) -> 'Game':
        """
        Returns:
            a game instance with the same initial parameters in its initial state
        """
        raise NotImplementedError

    def to_next_state(self, action: int) -> 'Game':
        """Return a copy of the game with the given action applied.

        Automatically updates current turn and game outcome for the new game instance, and the current player is
        switched to the opponent.

        Input:
            action: action taken by current player

        Returns:
            a copy of the game with the given action applied
        """
        raise NotImplementedError

    def to_planes(self) -> np.ndarray:
        """Converts the current board to the input of the neural network.

        Returns:
            a stack of (M + L) x N1 x N2 board-like planes representing various features of the board
            where N1, N2 are the board's dimensions, M is the number of features and L is the number of
            constant valued planes representing the state of hidden variables and special rules
        """
        raise NotImplementedError

    def valid_actions_mask(self) -> np.ndarray:
        """
        Returns:
            validMoves: a binary vector of length self.max_actions, 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        raise NotImplementedError

    def symmetries(self, board_like: np.ndarray) -> list[np.ndarray]:
        """Get all isomorphic forms (symmetries + translations) of board pl

        Use this method on board planes and policy planes only.

        Input:
            board_like: a ndarray with the same number of dimensions as planes_shape or policy_shape,
                        and the last two dimensions are identical to board_shape

        Returns:
            a list of ndarrays representing the isomorphic forms of board_like
        """
        raise NotImplementedError

    def random_symmetry(self, board_like: np.ndarray) -> np.ndarray:
        """Get a random symmetric form of the board.

        Call this function prior to inference in mcts to average out bias

        Input:
            board_like: a ndarray with the same number of dimensions as planes or policy_shape,
                        and the last two dimensions are identical to board_shape

        Returns:
            a ndarray representing the isomorphic form of board_like
        """
        raise NotImplementedError

    def score(self) -> int:
        """
        Returns:
            an integer representing the score of the current player
        """
        raise NotImplementedError

    def move_to_action(self, move: tuple | list) -> int:
        """Converts a move tuple to an action integer.

        Input:
            move: a tuple or list (Move, row, col) representing the move

        Returns:
            an integer representing the action
        """
        raise NotImplementedError

    def action_to_move(self, action: int) -> tuple[Any, int, int]:
        """Converts an action integer to a move tuple.

        Input:
            action: an integer representing the action

        Returns:
            a tuple of integers (Move, row, col) representing the move
        """
        raise NotImplementedError

    def render(self):
        """Renders the current board to stdout."""
        raise NotImplementedError
