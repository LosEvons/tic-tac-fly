from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tic_tac_fly.brain.reservoir import Reservoir

# Summaries of the state, structure and activity of the reservoir network.

@dataclass(frozen=True)
class NetworkStats:
    """Fixed properties of a reservoir network"""    
    n_neurons: int
    density: float
    inhibitory_fraction: float
    raw_spectral_radius: float
    spectral_radius: float
    operator_norm: float
    contraction_guaranteed: bool
    orphan_neurons: int
    
    @classmethod
    def of(cls, reservoir: Reservoir) -> NetworkStats:
        connectome = reservoir.connectome
        return cls(
            n_neurons=reservoir.n_neurons,
            density=connectome.density,
            inhibitory_fraction=connectome.inhibitory_fraction,
            raw_spectral_radius=reservoir.raw_spectral_radius,
            spectral_radius=float(np.max(np.abs(reservoir.eigenvalues))),
            operator_norm=reservoir.operator_norm,
            contraction_guaranteed=reservoir.contraction_guaranteed,
            orphan_neurons=reservoir.orphan_neurons,
        )
        

@dataclass(frozen=True)
class NetworkStateStats:
    """Summary of a single reservoir state vector"""
    mean_activity: float
    mean_abs: float
    norm: float
    saturated: float    # fraction of neurons where |x| > 0.9; respond pretty much the same to every board input
    desaturated: float  # fraction of neurons where |x| < 0.05; don't carry too much signal
    max_abs: float
    
    @classmethod
    def of(cls, state: np.ndarray) -> NetworkStateStats:
        magnitude = np.abs(state)
        return cls(
            mean_activity=float(state.mean()),
            mean_abs=float(magnitude.mean()),
            norm=float(np.linalg.norm(state)),
            saturated=float((magnitude > 0.9).mean()),
            desaturated=float((magnitude < 0.05).mean()),
            max_abs=float(magnitude.max())
        )