import os
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import RidgeClassifier

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "flywire_eb_adjacency.npy"

def fetch_flywire_subgraph(
    output_path: Path = DATA_PATH,
    roi: str = "EB",
    max_neurons: int = 500,
) -> np.ndarray:
    # Downloads fruitfly connectome synapse data, and caches it locally as an np array
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join( # To make this path operation work on WSL due to a quirk with how navis works
        entry for entry in original_path.split(os.pathsep) if not entry.startswith("/mnt/") 
    )
    try:
        import fafbseg
    except ImportError as e:
        raise ImportError(
            "fafbseg is required to fetch FlyWire data!"
        ) from e
    finally:
        os.environ["PATH"] = original_path

    token = os.environ.get("FLYWIRE_TOKEN")
    if not token:
        raise RuntimeError(
            "FLYWIRE_TOKEN is not set. Add it to a .env file in the project root."
        )
    fafbseg.flywire.set_chunkedgraph_secret(token, overwrite=True)

    # Retrieve up to max_neurons number of neurons by ROI.
    annotations = fafbseg.flywire.search_annotations(roi)
    unique_neurons = annotations["root_id"].unique()
    selected_neurons = unique_neurons[:max_neurons]

    # Retrieve synaptic connection counts between selected neurons as an adjacency matrix W_raw
    adjacency_matrix = fafbseg.flywire.get_adjacency(selected_neurons)
    W_raw = adjacency_matrix.to_numpy(dtype=float)
    
    np.save(output_path, W_raw)
    print(f"Cached biological adjacency matrix to {output_path}")
    return W_raw

class FlyReservoir:
    def __init__(
        self,
        adjacency_matrix_path: Path = DATA_PATH,
        seed: int = 42  
    ):
        rng = np.random.default_rng(seed)
        
        if not adjacency_matrix_path.exists():
            print(f"{adjacency_matrix_path.name} not found. Fetching from FlyWire...")
            W_syn = fetch_flywire_subgraph(output_path=adjacency_matrix_path)
        else:
            W_syn = np.load(adjacency_matrix_path)
    
        self.n = W_syn.shape[0]
        
        # Could also be queried from the dataset, but for now I use an approximation
        syn_signs = rng.choice(
            [1.0, -1.0],
            size=W_syn.shape,
            p=[0.8, 0.2] # Excitatory | Inhibitory ratio. Biased towards more mammal brains I think, but quite standard. A closer ratio can be found in Dorkenwald et al. (2024)
        )
        self.W_bio = W_syn * syn_signs
        
        # Spectral radius normalization
        eigenvalues = np.linalg.eigvals(self.W_bio)
        self.max_eigen = np.max(np.abs(eigenvalues)) + 1e-9
        
        # Sparse sensory affarents (board (9 squares) to input neurons)
        self.W_in = rng.normal(0.0, 1.0, (self.n, 9))
        mask = rng.random(self.W_in.shape) < 0.15
        self.W_in = self.W_in * mask
    
    def simulate(
        self,
        board: np.ndarray,
        spectral_radius: float = 0.95,
        leak_rate: float = 0.35,
        simulation_steps: int = 25
    ):
        W = (self.W_bio / self.max_eigen) * spectral_radius
        x = np.zeros(self.n)
        for _ in range(simulation_steps):
            play = W @ x + self.W_in @ board
            state = np.tanh(play)
            x = (1.0 - leak_rate) * x + leak_rate * state
        
        return x
    
def train_baseline_readout(
    reservoir: FlyReservoir,
    samples: int = 1200,
    seed: int = 42
) -> RidgeClassifier:
    rng = np.random.default_rng(seed)
    X_train_boards = []
    y_train_moves = []
    
    for _ in range(samples):
        b = rng.choice([0, 1, -1], size=9, p=[0.5, 0.3, 0.2])
        emptys = np.where(b == 0)[0]
        if len(emptys) > 0:
            # Simple heuristic: center first, then random legal square
            target = 4 if 4 in emptys else rng.choice(emptys)
            X_train_boards.append(b)
            y_train_moves.append(target)
    
    # Simulate boards (project into the biological states)
    X_train_states = np.array([
        reservoir.simulate(
            b, spectral_radius=0.95, leak_rate=0.35, simulation_steps=25
        ) for b in X_train_boards
    ])
    
    # Fit linear regression layer
    readout = RidgeClassifier(alpha=1.0)
    readout.fit(X_train_states, y_train_moves)
    return readout