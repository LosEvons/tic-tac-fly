"""Linear readouts. Predict move value from network state.
This is essentially an interpreter for our fly's thinking.
"""


from __future__ import annotations

import numpy as np

from sklearn.linear_model import RidgeCV

from tic_tac_fly.game.board import BOARD_SIZE, Board

class SquareReadout:
    """One linear ridge regressor per square that predicts O's minmax value"""
    
    def __init__(self, alphas: np.ndarray | None = None):
        """Alphas are candidates for ridge penalties, that help keep weights small. Defaults to 25 log-spaced values between 1e-6 and 10"""
        self.alphas = np.logspace(-6, 1, 25) if alphas is None else alphas
        self.models: list[RidgeCV] = []
        self.fitted = False
        
    def _fit_to_square(
        self, 
        square: int, 
        states: np.ndarray, 
        boards: np.ndarray,
        targets: np.ndarray
        ) -> RidgeCV:
        """Fit the model for a given square, if the square is empty in any board."""
        mask = boards[:, square] == 0
        if not mask.any():
            raise ValueError(f"No boards with square {square} empty. Cannot initialize RidgeCV for square {square}.")
        return RidgeCV(alphas=self.alphas).fit(states[mask], targets[mask, square])
    
    def fit(
        self,
        states: np.ndarray, 
        boards: np.ndarray,
        targets: np.ndarray
        ) -> SquareReadout:
        """Fit 9 RidgeCV models for 9 squares"""
        self.models = [
            self._fit_to_square(square, states, boards, targets)
            for square in range(BOARD_SIZE)
        ]
        self.fitted = True
        return self
    
    def predict(self, states: np.ndarray) -> np.ndarray:
        """Returns the predicted move values for all squares of all states."""
        if not self.fitted:
            raise RuntimeError(f"Readout not fitted when trying to predict.")
        states = np.atleast_2d(states)
        return np.column_stack(
            [model.predict(states) for model in self.models]
        )
        
    def score_position(self, state: np.ndarray) -> np.ndarray:
        """Predicted values for all squares of a single state"""
        return self.predict(state.reshape(1, -1))[0]
    
    def best_square(self, state: np.ndarray, board: Board) -> int:
        """Returns a legal square with the highest predicted value"""
        is_legal = board.legal_mask()
        if not is_legal.any():
            raise ValueError("No legal moves when trying to fetch best square from readout")
        return int(
            np.argmax(np.where(
                is_legal, self.score_position(state), -np.inf # -inf on occupied squares ensures argmax never picks them
            ))
        )
    
    @property
    def chosen_alphas(self) -> list[float]:
        """The alpha (ridge penalty) chosen by cross-validation for each square."""
        return [float(model.alpha_) for model in self.models]