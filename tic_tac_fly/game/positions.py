"""Representations for every position the fly can face with move histories. Used for training.
The fly is trained on every reachable unfinished position on its turn.
Each position is paired with several possible move orders that lead to it. This teaches the readout to recognize a position independent of any one way it was reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, chain, repeat, takewhile, zip_longest
import math
from typing import Iterator

import numpy as np

from tic_tac_fly.game.board import Board, Game, Move, Player
from tic_tac_fly.game.minimax import MinimaxSolver, best_squares


def successors(board: Board) -> Iterator[Board]:
    """Every new board produced by a legal move on given board"""
    turn = board.turn()
    return (
        board.play(int(square), turn)
        for square in board.legal_moves()
        )
    
def _next_layer(boards: tuple[Board, ...]) -> tuple[Board, ...]:
    """All distinct boards produced by a legal move on any given board in boards.
    Essentially a loop on successors. 
    """
    c = chain.from_iterable(
        successors(board) for board in boards if not board.is_terminal
    )
    return tuple(
        sorted( # sorted by raw bytes for reproducibility between runs
            set(c), # set merges board transpositions (same board, different move history)
            key=lambda board: board.cells.tobytes()
        )
    )
    
def generate_layers() -> Iterator[tuple[Board, ...]]:
    """Generate and group boards by number of moves played exhaustively starting from an empty board."""
    return takewhile(
        bool,
        accumulate(
            repeat(None),
            lambda boards, _: _next_layer(boards),
            initial=(Board(),)
        )
    )

def reachable_positions(actor: Player = Player.O) -> list[Board]:
    """Every reachable unfinished position where it's actor's turn."""
    return [
        board
        for board in chain.from_iterable(generate_layers())
        if not board.is_terminal and board.turn() is actor
    ]
    
def _alternate(x_squares: np.ndarray, o_squares: np.ndarray) -> tuple[Move, ...]:
    """Alternating X, O moves with X starting into a move sequence"""
    return tuple(
        move
        for move_pair in zip_longest
        (
            (Move(int(square), Player.X) for square in x_squares),
            (Move(int(square), Player.O) for square in o_squares),
        )
        for move in move_pair
        if move is not None
    )
    
def move_ordering(board: Board, count: int, rng: np.random.Generator) -> list[tuple[Move, ...]]:
    """Get up to count unique move histories for a board state."""
    X_squares, O_squares = board.squares_of(Player.X), board.squares_of(Player.O)
    n_unique = math.factorial(len(X_squares)) * math.factorial(len(O_squares)) # number of total unique move histories for board
    seen: dict[tuple[Move, ...], None] = {} # dictionary drops repeated histories
    while len(seen) < min(count, n_unique):
        seen.setdefault(
            _alternate(rng.permutation(X_squares), rng.permutation(O_squares)), None
        )
    return list(seen)
    

    

@dataclass(frozen=True)
class PositionSet:
    """Positions, minmax targets of positions, and sampled games reaching each position"""
    boards: list[Board]
    targets: np.ndarray
    games: list[Game]
    game_index: np.ndarray
    
    @classmethod
    def generate_exhaustively(
        cls,
        solver: MinimaxSolver | None = None,
        orderings_per_board: int = 3, seed: int = 0
        ) -> PositionSet:
        """Every position where it is the solver's player's turn."""
        solver = solver or MinimaxSolver()
        boards = reachable_positions(solver.maximiser)
        rng = np.random.default_rng(seed)
        sampled_games = [
            (i, Game.from_moves(moves))
            for i, board in enumerate(boards)
            for moves in move_ordering(board, orderings_per_board, rng)
        ]
        i, games = zip(*sampled_games)
        return cls(
            boards=boards,
            targets=np.array([solver.training_targets(board) for board in boards]),
            games=list(games),
            game_index=np.array(i)
        )
    
    def subset(self, positions: np.ndarray) -> PositionSet:
        """PositionSet with games ending in given positions"""
        positions = np.asarray(positions)
        reindex = {int(old): new for new, old in enumerate(positions)}
        kept_games = [game for game, i in enumerate(self.game_index) if int(i) in reindex]
        return PositionSet(
            boards=[self.boards[i] for i in positions],
            targets=self.targets[positions],
            games=[self.games[game] for game in kept_games],
            game_index=np.array([reindex[int(self.game_index[game])] for game in kept_games])
        )
        
    def split(self, test_fraction: float = 0.2, seed: int = 0) -> tuple[PositionSet, PositionSet]:
        """Split PositionSet into (train, test) sets."""
        ord = np.random.default_rng(seed).permutation(len(self.boards))
        n_test = int(round(test_fraction * len(ord)))
        return self.subset(np.sort(ord[n_test:])), self.subset(np.sort(ord[:n_test]))
        
    @property
    def game_boards(self) -> np.ndarray:
        return np.array([self.boards[i].cells for i in self.game_index])
    
    @property
    def game_minmax_targets(self) -> np.ndarray:
        return self.targets[self.game_index]
    
    def optimal_squares(self, pos: int, tolerance: float = 1e-9) -> frozenset[int]:
        return best_squares(self.targets[pos], self.boards[pos].legal_mask(), tolerance)
        
    def __len__(self) -> int:
        """Number of games. Not positions."""
        return len(self.games)