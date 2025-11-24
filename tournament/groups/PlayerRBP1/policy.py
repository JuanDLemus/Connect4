import math
import os
import time
import gzip
import pickle
from typing import Dict, Tuple, List

import numpy as np

from connect4.policy import Policy
from connect4.connect_state import ConnectState


# --- Utilidades de tablero y MCTS ---
def apply_move(state: np.ndarray, col: int, player: int) -> np.ndarray:
    board = state.copy()
    for r in range(board.shape[0] - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return board
    return board


def check_win(board: np.ndarray, player: int) -> bool:
    rows, cols = board.shape

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            window = board[r, c : c + 4]
            if np.all(window == player):
                return True

    # Vertical
    for c in range(cols):
        for r in range(rows - 3):
            window = board[r : r + 4, c]
            if np.all(window == player):
                return True

    # Diagonal
    for r in range(rows - 3):
        for c in range(cols - 3):
            if (
                board[r, c] == player
                and board[r + 1, c + 1] == player
                and board[r + 2, c + 2] == player
                and board[r + 3, c + 3] == player
            ):
                return True

    # Diagonal
    for r in range(3, rows):
        for c in range(cols - 3):
            if (
                board[r, c] == player
                and board[r - 1, c + 1] == player
                and board[r - 2, c + 2] == player
                and board[r - 3, c + 3] == player
            ):
                return True

    return False


def is_bad_move(state: np.ndarray, col: int, player: int) -> bool:
    temp = state.copy()
    landing_row = -1
    for r in range(temp.shape[0] - 1, -1, -1):
        if temp[r, col] == 0:
            temp[r, col] = player
            landing_row = r
            break
    if landing_row <= 0:
        return False

    opponent = -player
    temp[landing_row - 1, col] = opponent
    return check_win(temp, opponent)


def fast_rollout(state: np.ndarray, player: int, max_depth: int = 15) -> int:
    board = state.copy()
    current = player

    for _ in range(max_depth):
        valid = [c for c in range(board.shape[1]) if board[0, c] == 0]
        if not valid:
            return 0

        col = int(np.random.randint(len(valid)))
        col = valid[col]

        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == 0:
                board[r, col] = current
                break

        if check_win(board, current):
            return current

        current = -current

    return 0


class _Node:
    def __init__(self, state: np.ndarray, player: int, parent=None, action: int | None = None) -> None:
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children: List["_Node"] = []
        self.untried: List[int] = [c for c in range(state.shape[1]) if state[0, c] == 0]
        self.wins: float = 0.0
        self.visits: int = 0

    def expand(self) -> "_Node":
        col = self.untried.pop()
        next_state = apply_move(self.state, col, self.player)
        child = _Node(next_state, -self.player, parent=self, action=col)
        self.children.append(child)
        return child

    def best_child(self, c_param: float) -> "_Node":
        best = None
        best_val = -float("inf")
        log_n = math.log(self.visits) if self.visits > 0 else 0.0

        for child in self.children:
            if child.visits == 0:
                return child

            exploit = child.wins / child.visits
            explore = 0.0
            if log_n > 0.0:
                explore = c_param * math.sqrt(log_n / child.visits)

            score = exploit + explore
            if score > best_val:
                best_val = score
                best = child

        return best if best is not None else self.children[0]

    def update(self, reward: float) -> None:
        self.visits += 1
        self.wins += reward


def _mcts_search(root: _Node, player: int, time_limit: float, c_param: float) -> None:
    start = time.time()

    while True:
        for _ in range(50):
            node = root

            # Selección
            while not node.untried and node.children:
                node = node.best_child(c_param)

            # Expansión
            if node.untried:
                node = node.expand()

            # Simulación
            winner = fast_rollout(node.state, node.player)

            # Backpropagation
            cur = node
            while cur.parent is not None:
                move_maker = cur.parent.player
                if winner == move_maker:
                    reward = 1.0
                elif winner == -move_maker:
                    reward = 0.0
                else:
                    reward = 0.5
                cur.update(reward)
                cur = cur.parent

            root.visits += 1

        if time.time() - start > time_limit:
            break


def run_mcts(root_state: np.ndarray, player: int, time_limit: float, c_param: float = math.sqrt(2.0)) -> int:
    root = _Node(root_state, player)

    _mcts_search(root, player, max(time_limit, 0.01), c_param)

    if not root.children:
        valid = [c for c in range(root_state.shape[1]) if root_state[0, c] == 0]
        return valid[0] if valid else 0

    best = max(root.children, key=lambda ch: ch.visits)
    return int(best.action if best.action is not None else 0)


# --- Política híbrida ---
class RandomBeaterPro(Policy):
    def __init__(self, training_mode: bool = True, mcts_time_limit: float = 0.5, rl_action_prob: float = 0.5) -> None:
        self.training_mode = training_mode
        self.mcts_time_limit = float(mcts_time_limit)
        self.rl_action_prob = float(rl_action_prob)

        # Parámetros de Q-learning
        self.learning_rate = 0.1
        self.discount_factor = 0.99

        # Exploración adicional (opcional)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

        # Q-table y contadores para UCB1
        self.q_table: Dict[Tuple[bytes, int], float] = {}
        self.N_s: Dict[bytes, int] = {}
        self.N_sa: Dict[Tuple[bytes, int], int] = {}

        # Almacenamiento
        self.q_table_filename = "q_table.pkl.gz"
        self.q_table_path = os.path.join(os.path.dirname(__file__), self.q_table_filename)

        # RNG para decisiones aleatorias
        self.rng = np.random.default_rng()

    # --- Utilidades internas de RL ---

    def _get_state_key(self, board: np.ndarray) -> bytes:
        return board.tobytes()

    def _available_cols(self, board: np.ndarray) -> List[int]:
        return [c for c in range(board.shape[1]) if board[0, c] == 0]

    def _deduce_current_player(self, board: np.ndarray) -> int:
        cnt_red = int(np.count_nonzero(board == -1))
        cnt_yellow = int(np.count_nonzero(board == 1))
        return -1 if cnt_red == cnt_yellow else 1

    def _try_immediate_win_or_block(self, board: np.ndarray) -> int | None:
        player = self._deduce_current_player(board)
        opponent = -player
        avail = self._available_cols(board)

        cs = ConnectState(board, player)

        # Intento de victoria inmediata
        for col in avail:
            try:
                nxt = cs.transition(col)
            except Exception:
                continue
            if nxt.get_winner() == player:
                return int(col)

        # Bloqueo de victoria inmediata del oponente
        for col in avail:
            try:
                nxt = cs.transition(col)
            except Exception:
                continue

            opp_avail = nxt.get_free_cols()
            for opp_col in opp_avail:
                try:
                    opp_next = nxt.transition(opp_col)
                except Exception:
                    continue
                if opp_next.get_winner() == opponent:
                    return int(col)

        return None

    def next_is_final(self, board: np.ndarray) -> Tuple[int, List[int]]:
        rows, cols = board.shape
        candidate_cols: List[int] = []
        winning_player = 0

        for c in range(cols):
            if board[0, c] != 0:
                continue

            landing_row = None
            for r in range(rows - 1, -1, -1):
                if board[r, c] == 0:
                    landing_row = r
                    break
            if landing_row is None:
                continue

            for player in (-1, 1):
                new_board = board.copy()
                new_board[landing_row, c] = player
                state = ConnectState(new_board, player)
                if state.get_winner() == player:
                    candidate_cols.append(c)
                    winning_player = player
                    break

        return winning_player, candidate_cols

    def _select_action_ucb1(self, board: np.ndarray, candidates: List[int]) -> int:
        state_key = self._get_state_key(board)

        # Asegurarse de que el número total de visitas sea al menos 1
        total_visits = self.N_s.get(state_key, 0) + 1
        c_ucb = math.sqrt(2.0)

        best_a = None
        best_score = -float("inf")

        for col in candidates:
            q = self.q_table.get((state_key, col), 0.0)

            # Visitas de este par (estado, acción)
            n_sa = self.N_sa.get((state_key, col), 0)

            if n_sa == 0:
                # Priorizar acciones nunca probadas
                score = float("inf")
            else:
                # log(total_visits) >= 0 porque total_visits >= 1
                bonus = c_ucb * math.sqrt(math.log(total_visits) / n_sa)
                score = q + bonus

            if score > best_score:
                best_score = score
                best_a = col

        if best_a is None:
            best_a = candidates[0]

        return int(best_a)


    # --- Interfaz Policy ---

    def mount(self, *args, **kwargs) -> None:
        """
        Inicializa el agente. Admite opcionalmente un parámetro de timeout
        que puede ser pasado por el entorno del torneo, por ejemplo:
            policy.mount(POLICY_ACTION_TIMEOUT)
        """
        # Timeout opcional pasado por el entorno (Gradescope / torneo)
        if args:
            self.time_out = args[0]
        else:
            self.time_out = kwargs.get("time_out", None)

        # Si nos dan un timeout y no se ha fijado explícitamente mcts_time_limit,
        # lo aprovechamos como límite superior para MCTS en modo torneo.
        if self.time_out is not None:
            # Si ya tienes mcts_time_limit configurado desde fuera, respétalo;
            # en caso contrario, ajústalo en función del timeout.
            if not hasattr(self, "mcts_time_limit") or self.mcts_time_limit is None:
                self.mcts_time_limit = float(self.time_out) * 0.9
            else:
                # Garantiza que no excedas el timeout
                self.mcts_time_limit = min(float(self.time_out) * 0.9, float(self.mcts_time_limit))

        # Carga de Q-table si existe
        if os.path.exists(self.q_table_path):
            try:
                with gzip.open(self.q_table_path, "rb") as f:
                    self.q_table = pickle.load(f)
            except Exception:
                self.q_table = {}

        # En modo evaluación/tournament no queremos exploración epsilon-greedy
        if not self.training_mode:
            self.epsilon = 0.0


    def act(self, s: np.ndarray) -> int:
        board = s
        n_cols = board.shape[1]
        available = self._available_cols(board)
        if not available:
            return 0

        # Heurística directa: jugada que gana o bloquea de inmediato
        winner, cols = self.next_is_final(board)
        if winner != 0 and cols:
            return int(cols[0])

        immediate = self._try_immediate_win_or_block(board)
        if immediate is not None:
            return int(immediate)

        # Jugador actual según número de fichas en el tablero
        total_pieces = int(np.count_nonzero(board))
        player = 1 if total_pieces % 2 == 0 else -1

        valid_moves = [c for c in range(n_cols) if board[0, c] == 0]
        safe_moves = [m for m in valid_moves if not is_bad_move(board, m, player)]
        if not safe_moves:
            safe_moves = valid_moves

        if len(safe_moves) == 1:
            return int(safe_moves[0])

        # En modo entrenamiento, ocasionalmente dejar que RL seleccione la acción directamente
        use_rl = self.training_mode and self.q_table and (self.rng.random() < self.rl_action_prob)

        if use_rl:
            action = self._select_action_ucb1(board, safe_moves)
        else:
            # MCTS truncado por tiempo
            time_limit = max(self.mcts_time_limit, 0.01)
            action = run_mcts(board, player, time_limit)

            # Si MCTS devuelve algo extraño o inseguro, usar un fallback basado en RL/centro/random
            if action not in safe_moves:
                if self.q_table:
                    action = self._select_action_ucb1(board, safe_moves)
                else:
                    center = n_cols // 2
                    if center in safe_moves:
                        action = center
                    else:
                        action = int(self.rng.choice(safe_moves))

        return int(action)

    # --- Actualización de Q-learning ---

    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        s_key = self._get_state_key(state)
        ns_key = self._get_state_key(next_state)

        old = self.q_table.get((s_key, action), 0.0)

        next_avail = [c for c in range(next_state.shape[1]) if next_state[0, c] == 0]
        if not next_avail:
            next_max = 0.0
        else:
            next_max = max(self.q_table.get((ns_key, c), 0.0) for c in next_avail)

        new_val = old + self.learning_rate * (reward + self.discount_factor * next_max - old)
        self.q_table[(s_key, action)] = new_val

        # Contadores para UCB1
        self.N_s[s_key] = self.N_s.get(s_key, 0) + 1
        self.N_sa[(s_key, action)] = self.N_sa.get((s_key, action), 0) + 1

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save_q_table(self) -> None:
        try:
            with gzip.open(self.q_table_path, "wb") as f:
                pickle.dump(self.q_table, f)
        except Exception:
            pass
