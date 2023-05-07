from __future__ import print_function
import sys
from copy import deepcopy
from enum import Enum
from itertools import product

from flags import PlayerColour, GameOutcome

sys.path.append('..')
from Game import Game
import numpy as np


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
    def from_num(cls, num: int) -> 'Move':
        for move in cls:
            if move.num == num:
                return move
        raise IndexError(f"Move number {num} is not valid.")

    @classmethod
    def all_spreads(cls) -> tuple['Move', ...]:
        return tuple(move for move in cls if move != cls.SPAWN)


class InflexionGame(Game):
    def __init__(self, n: int,
                 first_mover: PlayerColour = PlayerColour.RED,
                 curr_player: PlayerColour = None,
                 board: np.ndarray = None,
                 curr_turn: int = 0,
                 max_turns: int = 100,
                 max_power: int = 6):
        max_actions_per_cell = 6 + 1
        policy_shape = max_actions_per_cell, n, n

        super().__init__(board_shape=(n, n), policy_shape=policy_shape, first_mover=first_mover)
        assert isinstance(n, int) and n > 0
        assert isinstance(curr_player, PlayerColour) or curr_player is None
        assert isinstance(board, np.ndarray) or board is None
        assert board is None or board.shape == (n, n)
        assert isinstance(curr_turn, int) and isinstance(max_turns, int) and isinstance(max_power, int)

        self._n = n
        if curr_player is not None:
            self._player = curr_player
        if board is None:
            self._board = np.zeros((n, n), dtype=int)
        else:
            self._board = board
        self._curr_turn = curr_turn
        self._max_turns = max_turns
        self._max_power = max_power

        self._planes_shape = 4, self._n, self._n
        self._max_power_at_spawn = 48
        # self.movesHistory = []

    def restarted(self) -> 'InflexionGame':
        return InflexionGame(self._n, first_mover=self._firstMover, max_turns=self._max_turns, max_power=self._max_power)

    def to_next_state(self, action: int) -> 'InflexionGame':
        assert isinstance(action, int) and 0 <= action < self.max_actions
        move = self.action_to_move(action)
        nextGame = deepcopy(self)
        nextGame.execute_move(move)
        assert nextGame._curr_turn == self._curr_turn + 1
        assert nextGame.player != self.player
        return nextGame

    def to_planes(self) -> np.ndarray:
        # return (self.player.num * self.board).reshape(self.boardStackShape)
        player_pieces = (self.player.owns(self._board)).astype(int)
        opponent_pieces = (self.player.opponent.owns(self._board)).astype(int)
        moves_count = np.ones(self.board_shape, dtype=int) * self._curr_turn
        can_spawn = np.tile(self.total_power() <= self._max_power_at_spawn, self.board_shape).astype(int)

        return np.stack([player_pieces, opponent_pieces, moves_count, can_spawn], axis=0)

    def valid_actions_mask(self) -> np.ndarray:
        valids = np.repeat(self.player.owns(self._board)[np.newaxis, ...], len(Move), axis=0)
        if self.total_power() <= self._max_power_at_spawn:
            valids[Move.SPAWN.num, :] = self._board == 0
        else:
            valids[Move.SPAWN.num, :] = False
        valids = valids.astype(int)
        return valids.ravel()

    def symmetries(self, board_like: np.ndarray) -> list[np.ndarray]:
        assert (isinstance(board_like, np.ndarray) and
                (board_like.shape == self._planes_shape or board_like.shape == self.policy_shape))

        isomorphic_shapes = [board_like.copy()]
        order_of_rotational_symmetry = 6

        rotated = [self.rotate(board_like, i) for i in range(1, order_of_rotational_symmetry)]
        translated = [self.translate(rotated_, j, axis='r') for rotated_ in rotated for j in range(1, self._n)]
        isomorphic_shapes += rotated + translated

        return isomorphic_shapes
    
    def random_symmetry(self, board_like: np.ndarray) -> np.ndarray:
        assert (isinstance(board_like, np.ndarray) and
                (board_like.shape == self._planes_shape or board_like.shape == self.policy_shape))

        orderOfRotationalSymmetry = 6
        board_like = self.rotate(board_like, np.random.randint(0, orderOfRotationalSymmetry))
        board_like = self.translate(board_like, np.random.randint(0, self._n), axis=np.random.choice(['r', 'q', 's']))
        return board_like

    def rotate(self, board_like: np.ndarray, k: int = 1) -> np.ndarray:
        """Returns a board rotated by 60 * k degrees

        Args:
            board_like: the board to be rotated
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
        assert (isinstance(board_like, np.ndarray) and len(board_like.shape) == len(self._planes_shape)
                and board_like.shape[-2:] == self.board_shape)
        assert isinstance(k, int)

        sign_flip = {0: np.array([1, 1, 1]),
                    1: np.array([-1, 1, 1]),
                    2: np.array([-1, 1, -1]),
                    3: np.array([-1, -1, -1]),
                    4: np.array([1, -1, -1]),
                    5: np.array([1, -1, 1])}
        order_of_symmetry = len(sign_flip)  # len(range(0, 360, 60))

        k %= order_of_symmetry
        r, q = np.indices(self.board_shape)
        s = (r + q) % self._n

        r, q, s = np.roll([r, q, s], k, axis=0) * sign_flip[k].reshape((-1, 1, 1))

        return board_like[:, r, q].copy()

    def translate(self, board_like: np.ndarray, shift: int, axis: str) -> np.ndarray:
        """Returns a board translated by shift cells along axis

        Args:
            board_like: the numpy array of shape (n, n, ...) to be translated
            shift: the number cells by which to shift the board
            axis: the axis along which to shift the board

        Returns:
            the translated boardLike array
        """
        assert (isinstance(board_like, np.ndarray) and len(board_like.shape) == len(self._planes_shape)
                and board_like.shape[-2:] == self.board_shape)
        assert isinstance(shift, int)
        assert axis in 'rqs'

        match axis:
            case 'r':
                # translation along r axis
                return np.roll(board_like, shift, axis=1)
            case 'q':
                # translation along q axis
                return np.roll(board_like, shift, axis=2)
            case 's':
                # translation along s axis
                return np.roll(np.roll(board_like, shift, axis=2), -shift, axis=1)
        raise ValueError("not supposed to reach here")

    def score(self) -> int:
        return self.piece_count_diff(self.player)

    def move_to_action(self, move: tuple | list) -> int:
        move_type, r, q = move
        assert 0 <= r < self._n and 0 <= q < self._n and move_type in Move
        return int(np.ravel_multi_index(([move_type.num], [r], [q]), self.policy_shape))

    def action_to_move(self, action: int) -> tuple[Move, int, int]:
        assert 0 <= action < self.max_actions
        move_num, r, q = np.unravel_index(action, self.policy_shape)
        move_type = Move.from_num(move_num)
        return move_type, r, q

    def render(self, ansi=False):
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
        for r, q in product(range(self._board.shape[0]), range(self._board.shape[1])):
            piece = self._board[r, q]
            if piece == 0:
                continue
            player = PlayerColour.from_piece(piece)
            power = abs(piece)
            board[(r, q)] = (player.token, power)

        dim = self._n
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

    def execute_move(self, move: tuple):
        """Execute a move on the board and update the game state."""
        move_type, r, q = move
        assert 0 <= r < self._n and 0 <= q < self._n and move_type in Move

        if move_type == Move.SPAWN and self.total_power() <= self._max_power_at_spawn:  # SPAWN
            assert self._board[r, q] == 0
            self._board[r, q] = self.player.num
        elif move_type in Move.all_spreads():  # SPREAD
            assert self.player.owns(self._board[r, q])
            power = np.abs(self._board[r, q])
            deltas = np.arange(1, power + 1).reshape(power, 1) * np.array(move_type.direction)
            index = (np.array((r, q)) + deltas) % self._n
            index = tuple(index.T)
            self._board[index] = np.abs(self._board[index]) + 1
            self._board[index] = np.where(self._board[index] > 6, 0, self._board[index]) * self.player.num
            self._board[r, q] = 0
        else:
            raise ValueError("Invalid move")

        # Update game status
        if (move_type in Move.all_spreads() and
                self._board[self.player.opponent.owns(self._board)].size == 0):
            self._outcome = GameOutcome.WON
        elif self._curr_turn >= self._max_turns:
            diff = self.power_diff(self.player)
            if diff >= 2:
                self._outcome = GameOutcome.WON
            elif diff <= -2:
                self._outcome = GameOutcome.LOST
            else:
                self._outcome = GameOutcome.DRAW
        elif (self._board == 0).all():
            self._outcome = GameOutcome.DRAW

        self._curr_turn += 1
        # self.movesHistory.append((self.player, move))
        self.player = self.player.opponent

    def power_diff(self, player: PlayerColour) -> int:
        """Count the total power difference for the given player"""
        assert player in PlayerColour
        total = self._board.sum()
        adjusted = player.num * total
        return adjusted

    def piece_count_diff(self, player: PlayerColour) -> int:
        """Count the # pieces difference for the given player"""
        assert player in PlayerColour
        diff = self._board[player.owns(self._board)].size - self._board[player.opponent.owns(self._board)].size
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

    def total_power(self):
        return np.sum(np.abs(self._board))
