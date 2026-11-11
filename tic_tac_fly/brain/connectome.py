from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# In Drosophila acetylcholine excites, GABA inhibits and glutamate inhibits
INHIBITORY_TRANSMITTERS = frozenset({"gaba", "glutamate"})

DEFAULT_ADJACENCY = Path("flywire_eb_adjacency.npy")
DEFAULT_SIGNS = Path("flywire_eb_signs.npy")
DEFAULT_ROOT_IDS = Path("flywire_eb_root_ids.npy")
DEFAULT_CELL_TYPES = Path("flywire_eb_cell_types.npy")

# -----------------------------------------


@dataclass(frozen=True)
class Connectome:
    """A signed wiring diagram between neurons. By default this uses the ellipsoid body (EB), which is responsible for the Fly's sense of direction.
    
    Attributes:
        synapses: (n, n) synapse counts of synapses[i, j] = number of synapses from presynaptic neuron j to postsynaptin neuron i.
        signs: (n,) where excitatory neurons are +1 and inhibitory neurons are -1 per neuron. Following Dale's principle we assume that
            a neuron releases the same neurotransmitter at all of its synapses.
        root_ids: FlyWire's neuron IDs.
        predicted_signs: True if signs came from FlyWire's predictions. False if they were assigned at random.
    """
    synapses: np.ndarray
    signs: np.ndarray
    root_ids: np.ndarray | None = None
    predicted_signs: bool = False
    cell_types: np.ndarray | None = None

    @property
    def n_neurons(self) -> int:
        """Number of neurons"""
        return self.synapses.shape[0]

    @property
    def inhibitory_fraction(self) -> float:
        """Fraction of neurons with a negative sign"""
        return float((self.signs < 0).mean())

    @property
    def density(self) -> float:
        """Fraction of neuron pairs with at least one synapse"""
        return float((self.synapses != 0).mean())

    @property
    def weights(self) -> np.ndarray:
        """Signed weight matrix W_bio. W_bio=synapse count * sign of sender"""
        return self.synapses * self.signs[None, :]

    @classmethod
    def random_signs(
        cls, synapses: np.ndarray, inhibitory_fraction: float = 0.7, seed: int = 42, root_ids: np.ndarray | None = None
    ) -> Connectome:
        """Wiring with each neuron's sign assigned at random."""
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            [1.0, -1.0],
            size=synapses.shape[0],
            p=[1 - inhibitory_fraction, inhibitory_fraction],
        )
        return cls(synapses, signs, root_ids, predicted_signs=False)

    @classmethod
    def load(
        cls,
        adjacency_path: Path = DEFAULT_ADJACENCY,
        signs_path: Path = DEFAULT_SIGNS,
        root_ids_path: Path = DEFAULT_ROOT_IDS,
        cell_types_path: Path = DEFAULT_CELL_TYPES,
        **fallback,
    ) -> Connectome:
        """Load a saved connectome and fall back to random signs if no file for signs exists."""
        synapses = np.load(adjacency_path)
        root_ids = np.load(root_ids_path) if root_ids_path.exists() else None
        cell_types = np.load(cell_types_path) if cell_types_path.exists() else None
        if signs_path.exists():
            return cls(synapses, np.load(signs_path), root_ids, predicted_signs=True, cell_types=cell_types)
        warnings.warn(
            f"{signs_path} not found. Signs assigned at random. Delete {adjacency_path} and rerun to download signs again."
        )
        return cls.random_signs(synapses, root_ids=root_ids, **fallback)

    @classmethod
    def load_or_fetch(
        cls,
        adjacency_path: Path = DEFAULT_ADJACENCY,
        signs_path: Path = DEFAULT_SIGNS,
        **fetch_kwargs,
    ) -> Connectome:
        """Load connectome from disk or download and save it from FlyWire if missing"""
        if adjacency_path.exists():
            return cls.load(adjacency_path, signs_path)
        print(
            f"adjacency matrix not found in path: {adjacency_path}. Fetching from FlyWire."
        )
        connectome = fetch_from_flywire(**fetch_kwargs)
        connectome.save(adjacency_path, signs_path)
        return connectome

    def save(
        self, adjacency_path: Path = DEFAULT_ADJACENCY, signs_path: Path = DEFAULT_SIGNS, root_ids_path: Path = DEFAULT_ROOT_IDS, cell_types_path: Path = DEFAULT_CELL_TYPES
    ) -> None:
        """Save synapses, signs and root_IDs as .npy files."""
        np.save(adjacency_path, self.synapses)
        np.save(signs_path, self.signs)
        if self.root_ids is not None:
            np.save(root_ids_path, self.root_ids)
        if self.cell_types is not None:
            np.save(cell_types_path, self.cell_types)

    def scrambled(self, seed: int = 0) -> Connectome:
        """Control network for comparison. Same synapse counts randomly rewired.
        Keeps density and distribution of synapses, but destroys the wiring pattern.
        """
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(self.synapses.ravel()).reshape(self.synapses.shape)
        return replace(self, synapses=shuffled)

    def __repr__(self) -> str:
        return (
            f"Connectome({self.n_neurons} neurons,"
            f"Density {self.density:.1%},"
            f"{self.inhibitory_fraction:.0%} inhibitory, "
            f"{'predicted' if self.predicted_signs else 'random'} signs)"
        )


# ------------------------------------------


def _import_fafbseg():
    """Import fafbseg with Windows paths removed from PATH
    A fix for navis tripping over PATH including Windows directories under /mnt/ at import time
    """
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        e for e in original_path.split(os.pathsep) if not e.startswith("/mnt/")
    )
    try:
        import fafbseg

        return fafbseg
    finally:
        os.environ["PATH"] = original_path


EB_CELL_TYPES = r"^(ER|ExR|EPG|EPGt|PEG|PEN|EL)"

def fetch_from_flywire(neuropil: str = "EB", cell_types: str = EB_CELL_TYPES, max_neurons: int = 500) -> Connectome:
    """Download the wiring and transmtter signs for a specific brain region. (for a specific neuropil subgraph)
    Requires FLYWIRE_TOKEN in the environment.
    Neuropil is the FlyWire region name.
    """
    from dotenv import load_dotenv
    load_dotenv()
    fafbseg = _import_fafbseg()
    # Deal with token for authentication to flywire api
    if not (token := os.environ.get("FLYWIRE_TOKEN")):
        raise RuntimeError("FLYWIRE_TOKEN is not set in environment variables")
    fafbseg.flywire.set_chunkedgraph_secret(token, overwrite=True)
    
    # Get candidate neurons by cell type.
    annotations = fafbseg.flywire.get_hierarchical_annotations()
    if (candidates := (
        annotations[annotations.cell_type.fillna("").str.match(cell_types)]
        .drop_duplicates("root_id")
        .set_index("root_id")
    )).empty:
        raise RuntimeError(f"No neurons match cell types {cell_types!r}")

    # Get the adjacency matrix (synapse counts between these neurons) for synapses inside neuropil
    adjacency = fafbseg.flywire.get_adjacency(
        candidates.index.to_numpy(), neuropils=neuropil, filtered=True, batch_size=100 # pyright: ignore[reportArgumentType] because of type inference mismatch
        )
    # Drop neurons that receive no synapses
    received = adjacency.sum(axis=0)
    ids = received[received > 0].nlargest(max_neurons).index
    adjacency = adjacency.loc[ids, ids]

    # fafbseg returns presynaptics ar rows and targets as columns
    # Transpose adjacency matrix to match Connectome schema (presynaptics as columns and targets as rows)
    synapses = adjacency.to_numpy(dtype=float).T.copy()

    transmitters = candidates.loc[ids, "top_nt"].fillna("").str.lower()
    signs = np.where(transmitters.isin(INHIBITORY_TRANSMITTERS), -1.0, 1.0)
    
    t_cell = candidates.loc[ids, "cell_type"].to_numpy(dtype=str)

    return Connectome(synapses, signs, root_ids=np.asarray(ids), predicted_signs=True, cell_types=t_cell)
