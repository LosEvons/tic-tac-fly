from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# In Drosophila acetylcholine excites, GABA inhibits and glutamate inhibits
INHIBITORY_TRANSMITTERS = frozenset({"gaba", "glutamate", "glut"})

DEFAULT_ADJACENCY = Path("flywire_eb_adjacency.npy")
DEFAULT_SIGNS = Path("flywire_eb_signs.py")

# -----------------------------------------


@dataclass(frozen=True)
class Connectome:
    # A signed diagram of synapse connections
    # synapses[i, j] (i=postsynaptic, j=presynaptic)
    # signs[j] (either +1 or -1, by Dale's principle release same transmitter from all synapses)
    synapses: np.ndarray
    signs: np.ndarray
    root_ids: np.ndarray | None = None

    @property
    def n_neurons(self) -> int:
        return self.synapses.shape[0]

    @property
    def inhibitory_fraction(self) -> float:
        return float((self.signs < 0).mean())

    @property
    def density(self) -> float:
        return float((self.synapses != 0).mean())

    @property
    def weights(self) -> np.ndarray:
        return self.synapses * self.signs[None, :]

    @classmethod
    def random_signs(
        cls, synapses: np.ndarray, inhibitory_fraction: float = 0.2, seed: int = 42
    ) -> Connectome:
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            [1.0, -1.0],
            size=synapses.shape[0],
            p=[1 - inhibitory_fraction, inhibitory_fraction],
        )
        return cls(synapses, signs)

    @classmethod
    def load(
        cls,
        adjacency_path: Path = DEFAULT_ADJACENCY,
        signs_path: Path = DEFAULT_SIGNS,
        **fallback,
    ) -> Connectome:
        synapses = np.load(adjacency_path)
        if signs_path.exists():
            return cls(synapses, np.load(signs_path))
        return cls.random_signs(synapses, **fallback)

    @classmethod
    def load_or_fetch(
        cls,
        adjacency_path: Path = DEFAULT_ADJACENCY,
        signs_path: Path = DEFAULT_SIGNS,
        **fetch_kwargs,
    ) -> Connectome:
        if adjacency_path.exists():
            return cls.load(adjacency_path, signs_path)
        print(
            f"adjacency matrix not found in path: {adjacency_path}. Fetching from FlyWire."
        )
        connectome = fetch_from_flywire(**fetch_kwargs)
        connectome.save(adjacency_path, signs_path)
        return connectome

    def save(
        self, adjacency_path: Path = DEFAULT_ADJACENCY, signs_path: Path = DEFAULT_SIGNS
    ) -> None:
        np.save(adjacency_path, self.synapses)
        np.save(signs_path, self.signs)

    def __repr__(self) -> str:
        return (
            f"Connectome({self.n_neurons}) neurons,"
            f"Density {self.density:.1%},"
            f"{self.inhibitory_fraction:.0%} ainhibitory,"
        )


# ------------------------------------------


def _import_fafbseg():
    # A fix for navis not being good at dealing with paths on WSL
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        e for e in original_path.split(os.pathsep) if not e.startswith("/mnt/")
    )
    try:
        import fafbseg

        return fafbseg
    finally:
        os.environ["PATH"] = original_path


def fetch_from_flywire(neuropil: str = "EB", max_neurons: int = 500) -> Connectome:
    # Download a neuropil subgraph with transmitter signs
    fafbseg = _import_fafbseg()
    # Deal with token for authentication to flywire api
    if not (token := os.environ.get("FLYWIRE_TOKEN")):
        raise RuntimeError("FLYWIRE_TOKEN is not set in environment variables")
    fafbseg.flywire.set_chunkedgraph_secret(token, overwrite=True)

    # Construct filter for neuropil tailored for the API
    roi = neuropil if neuropil.endswith(("_R", "_L")) else f"{neuropil}_R"
    criteria = fafbseg.flywire.NeuronCriteria(input_neuropils=roi)
    ids = np.sort(
        np.asarray(fafbseg.flywire.search_annotations(criteria)["root_id"].unique())  # pyright: ignore[reportAttributeAccessIssue] because of type inference mismatch
    )[:max_neurons]
    if ids.size == 0:
        raise RuntimeError(f"No neurons found with neuropil {roi!r}")

    # Get the adjacency matrix from annotations
    adjacency = fafbseg.flywire.get_adjacency(ids, neuropils=roi, filtered=True)  # pyright: ignore[reportArgumentType] because of type inference mismatch
    ids = np.asarray(adjacency.index)

    # Transpose adjacency matrix to match fafbseg schema (sources as ROWS and targets as COLS)
    synapses = adjacency.to_numpy(dtype=float).T.copy()

    # Get predicted neurotransmitters (whether a neuron is inhibitory or excitatory)
    predictions = fafbseg.flywire.get_transmitter_predictions(
        ids.tolist(), single_pred=True
    )
    signs = np.array(
        [
            -1.0
            if str(predictions[int(root)]).lower() in INHIBITORY_TRANSMITTERS
            else 1.0
            for root in ids
        ]
    )

    return Connectome(synapses, signs, root_ids=ids)
