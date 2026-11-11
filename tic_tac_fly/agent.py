"""The fly as a player. reservoir + readout with training and evaluation"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tic_tac_fly.brain.connectome import Connectome
from tic_tac_fly.brain.readout import SquareReadout
from tic_tac_fly.brain.reservoir import Reservoir, ReservoirConfig
from tic_tac_fly.game.board import Game, Move, Player
from tic_tac_fly.game.positions import PositionSet



@dataclass
class TrainingReport:
    """Result of training a readout"""
    n_games: int
    n_positions: int
    optimal_play_rate: float
    held_out_play_rate: float | None
    chosen_alphas: list[float]
    
    def __str__(self) -> str:
        held_out = ("" if self.held_out_play_rate is None else f", {self.held_out_play_rate:.1f}% optimal moves on held-out positions")
        return (f"{self.n_positions} positions / {self.n_games} games = {self.optimal_play_rate:.2f}% optimal moves in training{held_out}.")
    

class FlyAgent:
    """Plays for O. Runs the game through the reservoir and reads the result out."""
    def __init__(self, reservoir: Reservoir, readout: SquareReadout | None = None):
        self.reservoir = reservoir
        self.readout = readout or SquareReadout()
        
    @classmethod
    def from_connectome(cls, connectome: Connectome, config: ReservoirConfig | None = None) -> FlyAgent:
        return cls(Reservoir(connectome, config))
    
    # training
    
    def train_readout(self, positions: PositionSet, test_fraction: float = 0.2, seed: int = 0) -> TrainingReport:
        """Fit readout on (1 - test_fraction)% of the positions and score it on test_fraction% of the positions."""
        train, test = positions.split(test_fraction, seed)
        states = self.reservoir.encode_all(train.games)
        self.readout.fit(states, train.game_boards, train.game_minmax_targets)
        return TrainingReport(
            n_games=len(train),
            n_positions=len(train.boards),
            optimal_play_rate=self.evaluate_performance(train, states),
            held_out_play_rate=self.evaluate_performance(test) if len(test) else None,
            chosen_alphas=self.readout.chosen_alphas,
        )
        
    def evaluate_performance(self, positions: PositionSet, states: np.ndarray | None = None) -> float:
        """Percentage of games where the fly chose a minimax optimal square"""
        if states is None:
            states = self.reservoir.encode_all(positions.games)
        scores = self.readout.predict(states)
        hit_sum = sum(
            int(np.argmax(np.where(game.board.legal_mask(), scores[i], -np.inf)))
            in positions.optimal_squares(int(positions.game_index[i]))
            for i, game in enumerate(positions.games)
        )
        return 100.0 * hit_sum / len(positions)
    
    # play
    
    def decide(self, game: Game) -> Move:
        """The fly's move in game"""
        state = self.reservoir.encode(game)
        best = self.readout.best_square(state, game.board)
        return Move(best, Player.O)
    
    def play(self, game: Game) -> Game:
        """Returns a new game with the fly's move added"""
        return game.play(self.decide(game).square, Player.O)
    
    def raw_move_scores(self, game: Game) -> np.ndarray:
        """Readout value for each square"""
        scores = self.readout.score_position(self.reservoir.encode(game))
        return np.where(game.board.legal_mask(), scores, np.nan)