"""A fixed recurrent network as a reservoir.
Moves are fed in as brief stimuli one by one. The resulting activity echoes through the network, and the final state of the network describes the whole game.
I use the leaky-integrator echo state network update rule for each time step:
x <- (1 - a) * x + a * tanh(W x + W_in u). This is described in the readme/documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property, reduce
from itertools import accumulate
from typing import Iterable

import numpy as np

from tic_tac_fly.brain.connectome import Connectome
from tic_tac_fly.game.board import BOARD_SIZE, Game


@dataclass(frozen=True)
class ReservoirConfig:
    """A hashable configuration for the reservoir, so it can be keyed."""
    spectral_radius: float = 0.95
    leak_rate: float = 0.35
    steps_per_move: int = 4
    input_density: float = 0.15
    input_scale: float = 1.0
    seed: int = 42
    
    def key(self) -> tuple:
        return(
            self.spectral_radius,
            self.leak_rate,
            self.steps_per_move,
            self.input_density,
            self.input_scale,
            self.seed,
        )

    def with_(self, **changes) -> ReservoirConfig:
        """Return a copy with some settings changed"""
        return replace(self, **changes)
    

class Reservoir:
    """A leaky echo state network with a connectome as weights
    
    A biological perspective, since it's interesting to think about how all this translates:
        self.recurrent is the fly's neural wiring scaled by the spectral radius
        
        self.input_weights are the translation of board squares into a sensory input to a random subset of neurons 
    """
    def __init__(self, connectome: Connectome, config: ReservoirConfig | None = None):
        self.connectome = connectome
        self.config = config or ReservoirConfig()
        rng = np.random.default_rng(self.config.seed)
        weights = connectome.weights
        
        # Control gain from loops by rescaling the matrix to spectral radius
        # Spectral radius is the largest abs(eigenvalue), which represents roughly
        # how much an activity pattern grows or shrinks each time it loops inside the network.
        self.raw_spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(weights)))
        ) + 1e-9 # avoid division by zero for empty matrices
        self.recurrent = (weights / self.raw_spectral_radius) * self.config.spectral_radius
        
        # Create sensory afferents; gaussian weights kept with the probability of input_density 
        shape = (self.n_neurons, BOARD_SIZE)
        self.input_weights = (
            rng.normal(0.0, 1.0, shape) * (rng.random(shape) < self.config.input_density) * self.config.input_scale
        )
        
    
    @property
    def n_neurons(self) -> int:
        return self.connectome.n_neurons
    
    @property
    def resting_state(self) -> np.ndarray:
        return np.zeros(self.n_neurons)
    
    # for spectral diagnostics
    
    @cached_property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the rescaled recurrent matrix"""
        return np.linalg.eigvals(self.recurrent)
    
    @cached_property
    def operator_norm(self) -> float:
        """Largetst singular value of the rescaled matrix.
        Represents the most W can stretch any state vector in a single step.
        """
        return float(np.linalg.norm(self.recurrent, 2))
    
    @property
    def contraction_guaranteed(self) -> bool:
        """Check whether the echo state property is guaranteed, proving that the network is stable (old information fades at a guaranteed rate). 
        Lack of this guarantee doesn't mean the network isn't stable. It's just not proven.

        """
        return self.operator_norm < 1.0
        
    @cached_property
    def orphan_neurons(self) -> int:
        """Neurons without direct board input."""
        return int((np.abs(self.input_weights).sum(axis=1) == 0).sum())
    
    # operations
    
    def _step(self, drive: np.ndarray, state: np.ndarray) -> np.ndarray:
        """One-time step of the leaky update"""
        leak_rate = self.config.leak_rate
        # x <- (1 - a)x + a(tanh(Wx + W_in u))
        return (1 - leak_rate) * state + leak_rate * np.tanh(self.recurrent @ state + drive)
    
    def integrate_layer(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Hold one move's input u for steps_per_move steps."""
        drive = self.input_weights @ np.asarray(u, dtype=float)
        for _ in range(self.config.steps_per_move):
            state = self._step(drive, state)
        return state
    
    def forward(self, inputs: Iterable[np.ndarray]) -> np.ndarray:
        """Final state after integrating each input in turn from resting state."""
        return reduce(
            self.integrate_layer,
            inputs,
            self.resting_state
        )
                
    def encode(self, game: Game) -> np.ndarray:
        """Return the network state at the end of a game. (n_neurons,)"""
        return self.forward(game.inputs())
    
    def encode_all(self, games: Iterable[Game]) -> np.ndarray:
        """Return the network states at the end of all games provided. (n_games, n_neurons)"""
        return np.array(
            [self.encode(game) for game in games]
        )

    def trace(self, inputs: Iterable[np.ndarray]) -> np.ndarray:
        """State after each input starting from resting state."""
        return np.array(list(
            accumulate(
                inputs,
                self.integrate_layer,
                initial=self.resting_state
            )))
        
    def trace_steps(self, game: Game) -> np.ndarray:
        """State at every step of a game, starting at rest."""
        states = [self.resting_state]
        for u in game.inputs():
            drive = self.input_weights @ u
            for _ in range(self.config.steps_per_move):
                states.append(self._step(drive, states[-1]))
        return np.array(states)
        
    def trace_game(self, game: Game) -> np.ndarray:
        """Trace over all moves of a game"""
        return self.trace(game.inputs())
    
    def __repr__(self) -> str:
        return (f"Reservoir({self.n_neurons} neuron count), "
                f"spect.rad.={self.config.spectral_radius}, "
                f"leak rate ={self.config.leak_rate}, "
                f"layers ={self.config.steps_per_move})"
                )