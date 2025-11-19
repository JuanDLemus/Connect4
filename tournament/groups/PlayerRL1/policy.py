import numpy as np
import pickle
import os
import gzip
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import Tuple

# Trivial logger helper
def _log(msg: str) -> None:
    try:
        print(msg)
    except Exception:
        pass


class RLPolicy(Policy):
    """
    Política RL basada en Q-Learning con:
    - carga/guardado comprimido de q_table (.pkl.gz)
    - representación compacta de estado (board.tobytes())
    - heurística inmediata: jugar victoria inmediata o bloquear victoria del oponente
    """

    def __init__(self, training_mode: bool = False):
        self.training_mode = training_mode
        # Hiperparámetros por defecto (puedes afinarlos)
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01

        # Q-Table: clave -> (state_bytes, action), valor -> float
        self.q_table: dict[Tuple[bytes, int], float] = {}

        # Ruta del archivo comprimido
        self.q_table_filename = "q_table.pkl.gz"
        self.q_table_path = os.path.join(os.path.dirname(__file__), self.q_table_filename)

    def _get_state_key(self, board: np.ndarray) -> bytes:
        """Representación compacta del estado para usar como clave: bytes del array."""
        # usaremos board.tobytes() para obtener una representación compacta e inmutable
        return board.tobytes()

    def _deduce_current_player(self, board: np.ndarray) -> int:
        """
        Deducción del jugador que debe mover a partir del tablero:
        - Si cont_red == cont_yellow -> es el turno del jugador -1 (Rojo)
        - Si cont_red > cont_yellow -> es el turno del jugador 1 (Amarillo)
        """
        cnt_red = int(np.count_nonzero(board == -1))
        cnt_yellow = int(np.count_nonzero(board == 1))
        if cnt_red == cnt_yellow:
            return -1
        else:
            return 1

    def _available_cols(self, board: np.ndarray) -> list[int]:
        return [c for c in range(board.shape[1]) if board[0, c] == 0]

    def _try_immediate_win_or_block(self, board: np.ndarray) -> int | None:
        """
        Comprueba movimientos inmediatos:
        1) Si hay una acción que nos hace ganar ahora, devuélvela.
        2) Si hay una acción que bloquea una victoria inmediata del oponente, devuélvela.
        Si no hay, devuelve None.
        """
        player = self._deduce_current_player(board)
        opponent = -player
        avail = self._available_cols(board)

        # crear un ConnectState con el player deducido para usar transition()
        cs = ConnectState(board, player)

        # 1) buscar victoria inmediata propia
        for col in avail:
            try:
                next_state = cs.transition(col)
            except Exception:
                continue
            if next_state.get_winner() == player:
                return int(col)

        # 2) buscar bloqueo de victoria del oponente: simular que el oponente juega tras cada nuestro movimiento
        for col in avail:
            try:
                next_state = cs.transition(col)
            except Exception:
                continue
            # ahora el siguiente jugador es opponent en next_state
            opp_avail = next_state.get_free_cols()
            for opp_col in opp_avail:
                try:
                    opp_next = next_state.transition(opp_col)
                except Exception:
                    continue
                if opp_next.get_winner() == opponent:
                    # bloquear: si existe una jugada nuestra que evita que el oponente gane,
                    # preferimos esta jugada.
                    # Nota: Esto detecta situaciones donde sin hacer 'col' el oponente ganaría.
                    return int(col)
        return None

    def mount(self) -> None:
        """
        Intenta cargar la Q-Table comprimida desde:
         - la ruta local (cwd del módulo)
         - (intento seguro) recursos del paquete (importlib.resources) si está instalado como paquete
        Si no se encuentra, deja la q_table vacía y muestra mensaje.
        """
        loaded = False

        # Intentar cargar desde self.q_table_path (archivo junto al módulo)
        try:
            if os.path.exists(self.q_table_path):
                with gzip.open(self.q_table_path, "rb") as f:
                    self.q_table = pickle.load(f)
                _log(f"[RLPolicy] Q-Table cargada desde {self.q_table_path} ({len(self.q_table)} entradas).")
                loaded = True
        except Exception as e:
            _log(f"[RLPolicy] Error leyendo {self.q_table_path}: {e}")

        # Intento seguro con importlib.resources (solo si no se cargó aún)
        if not loaded:
            try:
                import importlib.resources as pkg_resources
                pkg = __package__  # debería ser 'groups.RL_Agent' si se importa como paquete
                if pkg is None:
                    raise RuntimeError("paquete no disponible para importlib.resources")
                candidate = pkg_resources.files(pkg).joinpath(self.q_table_filename)
                if candidate.is_file():
                    with gzip.open(candidate, "rb") as f:
                        self.q_table = pickle.load(f)
                    _log(f"[RLPolicy] Q-Table cargada via importlib.resources desde {candidate} ({len(self.q_table)} entradas).")
                    loaded = True
            except Exception:
                # no hacemos crash, solo informamos
                pass

        if not loaded:
            _log("[RLPolicy] No se encontró Q-Table en ruta local ni via importlib.resources. Se inicializa Q-Table vacía.")

        # Si no estamos en modo training, desactivar exploración
        if not self.training_mode:
            self.epsilon = 0.0

    def act(self, s: np.ndarray) -> int:
        """
        Decide una acción (columna) a tomar dado un estado del tablero s (np.ndarray).
        Flujo:
         - si existe jugada ganadora inmediata -> jugarla
         - si existe jugada que bloquea una victoria inmediata -> jugarla
         - si training_mode y rand < epsilon -> explorar (aleatorio)
         - si Q-Table no vacía -> elegir acción con mayor Q
         - si Q-Table vacía -> fallback aleatorio entre disponibles
        """
        
        next_final, cols = self.next_is_final(s)
        if next_final != 0 and cols:
            return int(cols[0])
        
        board = s
        avail = self._available_cols(board)
        if not avail:
            return 0

        # 1) heurística inmediata (win/block)
        immediate = self._try_immediate_win_or_block(board)
        if immediate is not None:
            return int(immediate)

        # 2) epsilon-greedy
        if self.training_mode and np.random.rand() < self.epsilon:
            return int(np.random.choice(avail))

        # 3) explotación con Q-Table (o fallback aleatorio si Q-Table vacía)
        state_key = self._get_state_key(board)
        n_cols = s.shape[1]
        center = n_cols // 2
        if (ConnectState.is_applicable(center, s)):
            return int(center)
        # si q_table vacía -> fallback a heurística simple (aleatorio)
        if not self.q_table:
            return int(np.random.choice(avail))

        q_values = {col: self.q_table.get((state_key, col), 0.0) for col in avail}
        max_q = max(q_values.values())
        best_cols = [c for c, q in q_values.items() if q == max_q]
        return int(np.random.choice(best_cols))

    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Actualiza la Q-Table usando la ecuación de Bellman."""
        s_key = self._get_state_key(state)
        ns_key = self._get_state_key(next_state)

        old = self.q_table.get((s_key, action), 0.0)

        # encontrar max para next state
        next_avail = [c for c in range(next_state.shape[1]) if next_state[0, c] == 0]
        if not next_avail:
            next_max = 0.0
        else:
            next_max = max([self.q_table.get((ns_key, c), 0.0) for c in next_avail])

        new_val = old + self.learning_rate * (reward + self.discount_factor * next_max - old)
        self.q_table[(s_key, action)] = new_val

    def decay_epsilon(self):
        """Reduce epsilon hacia el mínimo."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save_q_table(self):
        """Guarda la Q-Table comprimida con gzip en la ruta del módulo."""
        try:
            # Crear versión compacta opcional (sin podar por defecto)
            with gzip.open(self.q_table_path, "wb") as f:
                pickle.dump(self.q_table, f)
            _log(f"[RLPolicy] Q-Table guardada y comprimida en {self.q_table_path} ({len(self.q_table)} entradas).")
        except Exception as e:
            _log(f"[RLPolicy] Error guardando Q-Table: {e}")

    
    def next_is_final(self, s: np.ndarray) -> tuple[int, list[int]]:
        rows, cols = s.shape
        candidate_cols: list[int] = []
        winning_player = 0

        for c in range(cols):
            # Columna no jugable
            if s[0, c] != 0:
                continue

            # Fila donde caería la ficha en esta columna
            landing_row = None
            for r in range(rows - 1, -1, -1):
                if s[r, c] == 0:
                    landing_row = r
                    break
            if landing_row is None:
                continue

            # Simulamos la jugada para ambos jugadores (-1 y 1)
            for player in (-1, 1):
                new_board = s.copy()
                new_board[landing_row, c] = player
                state = ConnectState(new_board, player)
                if state.get_winner() == player:
                    candidate_cols.append(c)
                    winning_player = player
                    # No hace falta probar el otro jugador en esta columna
                    break

        return winning_player, candidate_cols
