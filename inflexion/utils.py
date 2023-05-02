# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion
from itertools import product

import numpy as np

from flags import PlayerColour
from inflexion.InflexionGame import InflexionGame


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
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{str}\033[0m"


def render_board(game: InflexionGame, ansi=False):
    """
    Visualise the Infexion hex board via a multiline ASCII string.
    The layout corresponds to the axial coordinate system as described in the
    game specification document.

    Example:

        >>> board = {
        ...     (5, 6): ("r", 2),
        ...     (1, 0): ("b", 2),
        ...     (1, 1): ("b", 1),
        ...     (3, 2): ("b", 1),
        ...     (1, 3): ("b", 3),
        ... }
        >>> print_board(board, ansi=False)

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
    assert isinstance(game, InflexionGame)

    board = {}
    for r, q in product(range(game.board.shape[0]), range(game.board.shape[1])):
        piece = game.board[r, q]
        if piece == 0:
            continue
        player = PlayerColour.from_piece(piece)
        power = abs(piece)
        board[(r, q)] = (player.token, power)

    dim = 7
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
                    output += apply_ansi(text, color=color, bold=False)
                else:
                    output += text
            else:
                output += " .. "
            output += "    "
        output += "\n"
    print(output)
