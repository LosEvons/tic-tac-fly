from __future__ import annotations

from enum import IntEnum
from functools import reduce

from typing import Iterator, NamedTuple, Sequence

import numpy as np

WINNING_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)

BOARD_SIZE = 9

_LINES = np.array(WINNING_LINES)
_SQUARES = np.array(BOARD_SIZE)


class Player(IntEnum):
    X = 1
    O = -1

    @property
    def other(self) -> Player:
        return Player.X if self is Player.O else Player.O

    @property
    def symbol(self) -> str:
        return "X" if self is Player.X else "O"


class Move(NamedTuple):
    square: int
    player: Player

    def one_hot(self) -> np.ndarray:
        # Input vector the reservoir consumes
        v = np.zeros(BOARD_SIZE)
        v[self.square] = int(self.player)
        return v


class Outcome(NamedTuple):
    winner: Player | None
    is_draw: bool

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or self.is_draw


class Board:
    """Immutable board position."""

    __slots__ = ("_cells",)

    def __init__(self, cells: Sequence[int] | np.ndarray | None = None):
        if cells is None:
            a = np.zeros(BOARD_SIZE, dtype=np.int8)
        else:
            a = np.asarray(cells, dtype=np.int8).reshape(BOARD_SIZE).copy()
        a.flags.writeable = False
        self._cells = a

    @property
    def cells(self) -> np.ndarray:
        return self._cells

    def vector(self) -> np.ndarray:
        return self._cells.astype(float)

    def __getitem__(self, square: int) -> int:
        return int(self._cells[square])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and bool((self._cells == other._cells).all())

    def __hash__(self) -> int:
        return hash(self._cells.tobytes())

    def __repr__(self) -> str:
        glyph = {0: "_", 1: "X", -1: "O"}
        r = (
            "".join(glyph[int(cell)] for cell in self._cells[i : i + 3])
            for i in (0, 3, 6)
        )
        return f"Board({'|'.join(r)})"

    @property
    def occupied(self) -> int:
        return int(np.count_nonzero(self._cells))

    def legal_moves(self) -> np.ndarray:
        return np.flatnonzero(self._cells == 0)

    def legal_mask(self) -> np.ndarray:
        return self._cells == 0

    def squares_of(self, player: Player) -> np.ndarray:
        return np.flatnonzero(self._cells == int(player))

    def turn(self) -> Player:
        n_X = int(np.count_nonzero(self._cells == Player.X))
        n_O = int(np.count_nonzero(self._cells == Player.O))
        if n_X == n_O:
            return Player.X
        if n_X == n_O + 1:
            return Player.O
        raise ValueError(f"Illegal position: n_X={n_X} & x_O={n_O}.")

    def play(self, square: int, player: Player | None = None) -> Board:
        """Return a new board with a square filled"""
        if self._cells[square] != 0:
            raise ValueError(f"{square} is occupied.")
        player = player if player is not None else self.turn()
        return Board(np.where(_SQUARES == square, int(player), self._cells))

    def outcome(self) -> Outcome:
        total = self._cells[_LINES].sum(axis=1)
        win = next(
            (player for player in Player if (total == 3 * int(player)).any()), None
        )
        return Outcome(win, win is None and self.occupied == BOARD_SIZE)

    @property
    def is_terminal(self) -> bool:
        return self.outcome().is_terminal


class Game:
    """A board with move history"""

    __slots__ = ("board", "moves")

    def __init__(self, board: Board | None = None, moves: Sequence[Move] | None = None):
        self.board = board if board is not None else Board()
        self.moves: tuple[Move, ...] = tuple(moves or ())

    @classmethod
    def from_moves(cls, moves: Sequence[Move]) -> Game:
        return reduce(
            lambda game, move: game.play(move.square, move.player), moves, cls()
        )

    def play(self, square: int, player: Player | None = None) -> Game:
        player = player if player is not None else self.board.turn()
        return Game(
            self.board.play(square, player), (*self.moves, Move(square, player))
        )

    def inputs(self) -> Iterator[np.ndarray]:
        return (move.one_hot() for move in self.moves)

    def __len__(self) -> int:
        return len(self.moves)

    def __repr__(self) -> str:
        return f"Game({len(self.moves)} layers, {self.board!r})"
