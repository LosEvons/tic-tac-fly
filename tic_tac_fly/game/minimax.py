"""A alpha-beta minimax scorer for legal moves.
Teaches perfect play. Scores produced are training targets for the readout to produce from reservoir states.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import accumulate
from typing import Iterable, Iterator, NamedTuple

import numpy as np

from tic_tac_fly.game.board import BOARD_SIZE, Board, Player


class ABWindow(NamedTuple):
    """A running state of an alpha-beta search at a node."""
    value: float
    alpha: float
    beta: float
    
    @property
    def cutoff(self) -> bool:
        """True when alpha >= beta, meaning remaining moves cannot change the result."""
        return self.alpha >= self.beta
    

def _last(items: Iterable[ABWindow]) -> ABWindow:
    """Run to the end of an iterator and return last element"""
    return deque(items, maxlen=1)[0] # consume the iterator, keeping the final element

def _until_cutoff(windows: Iterable[ABWindow]) -> Iterator[ABWindow]:
    """Yield ABWindows until the search window closes."""
    for window in windows:
        yield window
        if window.cutoff:
            return
        
def best_squares(values: np.ndarray, legals: np.ndarray, tolerance: float = 1e-9) -> frozenset[int]:
    """Legal squares with values (tied) for the best."""
    best = np.max(np.where(legals, values, -np.inf))
    return frozenset(
        int(square) for square in np.flatnonzero(legals & (values >= best - tolerance))
    )
    
        

@dataclass
class MinimaxSolver:
    """Alpha-beta search to score legal positions from O's POV
    Scores are in range [-1, 1], where 1 is a win for O.
    A depth penalty penalises slower wins, encourages slower losses.
    """
    maximiser: Player = Player.O
    depth_penalty: float = 0.01
    
    def terminal_value(self, board: Board) -> float | None:
        """Score of a finished position. None if position is not terminal."""
        outcome = board.outcome()
        if outcome.winner is self.maximiser:
            return 1.0 - self.depth_penalty * board.occupied
        if outcome.winner is not None:
            return -1.0 + self.depth_penalty * board.occupied
        return 0.0 if outcome.is_draw else None
    
    def search(self, board: Board, a: float = -np.inf, b: float = np.inf) -> float:
        """Gives a minimax value of a board when both sides play perfect"""
        terminal_value = self.terminal_value(board)
        if terminal_value is not None:
            return terminal_value
        turn = board.turn()
        maxing = turn is self.maximiser
        
        def eval_move(window: ABWindow, square: int) -> ABWindow:
            # Score child, then tighten the value and bound of this node
            c = self.search(board.play(int(square), turn), window.alpha, window.beta)
            if maxing:
                v = max(window.value, c)
                return ABWindow(v, max(window.alpha, v), window.beta)
            else:
                v = min(window.value, c)
                return ABWindow(v, window.alpha, min(window.beta, v))
        
        # thread the window through the moves one at a time until _until_cutoff stops it, at which point we take the final window value with _last
        initial_bounds = ABWindow(-np.inf if maxing else np.inf, a, b)
        windows = accumulate(
            board.legal_moves(),
            eval_move,
            initial=initial_bounds
        )
        return float(_last(_until_cutoff(windows)).value)
    
    def score_legal_moves(self, board: Board) -> np.ndarray:
        """The values of all legal moves by maximiser per square (np.nan when occupied)"""
        return np.array([
            self.search(board.play(square, self.maximiser))
            if board[square] == 0 else np.nan
            for square in range(BOARD_SIZE)
        ])
        
    def training_targets(self, board: Board) -> np.ndarray:
        """score_legal_moves with occupied squares zeroed (0.0).
        Used when training to make sure readout doesn't train on them.
        """
        return np.nan_to_num(self.score_legal_moves(board), nan=0.0)
    
    def best_moves(self, board: Board, tolerance: float = 1e-9) -> frozenset[int]:
        """Returns all squares, ties included, that are optimal moves for the maximiser"""
        v = self.score_legal_moves(board)
        return frozenset(
            int(score) for score in np.flatnonzero(v >= np.nanmax(v) - tolerance)
        )