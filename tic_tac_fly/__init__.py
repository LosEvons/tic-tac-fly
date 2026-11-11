"""A fly's brain playing tic-tac-toe."""
from importlib.metadata import version

from tic_tac_fly.agent import FlyAgent, TrainingReport
from tic_tac_fly.brain.connectome import Connectome, fetch_from_flywire
from tic_tac_fly.brain.diagnostics import NetworkStats, NetworkStateStats
from tic_tac_fly.brain.readout import SquareReadout
from tic_tac_fly.brain.reservoir import Reservoir, ReservoirConfig
from tic_tac_fly.game.board import Board, Game, Move, Outcome, Player
from tic_tac_fly.game.minimax import MinimaxSolver
from tic_tac_fly.game.positions import PositionSet

__version__ = version("tic-tac-fly")
 
__all__ = [
    "Board", "Game", "Move", "Outcome", "Player", "MinimaxSolver", "PositionSet",
    "Connectome", "fetch_from_flywire", "Reservoir", "ReservoirConfig", "SquareReadout",
    "NetworkStats", "NetworkStateStats", "FlyAgent", "TrainingReport",
]
 