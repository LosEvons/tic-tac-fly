import numpy as np

WINNING_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

def get_legal_moves(board: np.ndarray) -> np.ndarray:
    return np.where(board == 0)[0]

def check_whose_turn(board: np.ndarray) -> int:
    n_x = int(np.count_nonzero(board == 1))
    n_o = int(np.count_nonzero(board == -1))
    if n_x == n_o:
        return 1 # X turn
    if n_x == n_o + 1:
        return -1 # O turn
    raise ValueError(f"Illegal board when checking whose turn!")

def check_winner(board: np.ndarray) -> int | None:
    for line in WINNING_LINES:
        hits = int(board[list(line)].sum())
        if hits == 3:
            return 1
        if hits == -3:
            return -1
    return None

def check_draw(board: np.ndarray) -> bool:
    return check_winner(board) is None and not np.any(board == 0)

def check_is_terminal(board: np.ndarray) -> bool:
    return check_winner(board) is not None or check_draw(board)

def enumerate_o_moves() -> list[np.ndarray]:
    # Get all reachable non terminal boards on O's turn.
    # A tree DFS from an empty board. All possibilities (since tic-tac-toe doesn't have that many).
    seen: set[bytes] = set() # For deduplicating moves
    to_move: dict[bytes, np.ndarray] = {}
    
    def recurse(board: np.ndarray):
        k = board.tobytes()
        if k in seen:
            return
        seen.add(k)
        if check_is_terminal(board):
            return
        turn = check_whose_turn(board)
        if turn == -1:
            to_move[k] = board.copy()
        
        for move in get_legal_moves(board):
            next_state = board.copy()
            next_state[move] = turn
            recurse(next_state)
    
    recurse(np.zeros(9, dtype= int)) #np.int8
    return list(to_move.values())

def search(
    board: np.ndarray,
    alpha: float,
    beta: float
    ) -> float:
    depth = int(np.count_nonzero(board)) 
    winner = check_winner(board)
    if winner == -1:
        return 1.0 - 0.01 * depth # Prefer to win, but also to win faster. 0.01 is just what scales that preference
    if winner == 1:
        return -1.0 + 0.01 * depth # And same the other way around
    if check_draw(board):
        return 0.0
    
    turn = check_whose_turn(board)
    if turn == -1: # Maximize O's turn value
        value = -np.inf
        for move in get_legal_moves(board):
            next_state = board.copy()
            next_state[move] = -1
            value = max(value, search(next_state, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else: # Minimize X's turn value
        value = np.inf
        for move in get_legal_moves(board):
            next_state = board.copy()
            next_state[move] = 1
            value = min(value, search(next_state, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def minmax_move_values(board: np.ndarray) -> np.ndarray:
    # A vector of size 9 of minmax values (from O perspective)
    # Prefers fater wins and slower losses. Always win > draw > loss
    # Always compute each root from (-inf, inf) alpha-beta. This is crude, but the game being played right now is simple enough for it.
    values = np.full(9, np.nan, dtype=np.float64)
    for move in get_legal_moves(board):
        next_state = board.copy()
        next_state[move] = -1
        values[move] = search(next_state, alpha=-np.inf, beta=np.inf)
    return values
