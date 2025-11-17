import numpy as np
from connect4.policy import Policy


class MCTSPolicy(Policy):
    """
    Monte Carlo Tree Search (MCTS) para Connect4 usando solo el tablero (np.ndarray).

    - Estado del nodo: (board.tobytes(), player_to_move).
    - Selección: UCT / UCB1.
    - Rollout: aleatorio.
    """

    def mount(self) -> None:
        # Hiperparámetros
        self.simulations_per_move = 300   # puedes subir/bajar esto
        self.c_uct = float(np.sqrt(2.0))  # constante de exploración UCT

        # RNG
        self.rng = np.random.default_rng()

        # Árbol de búsqueda
        self.N = {}        # N[state_key]: visitas del estado
        self.N_sa = {}     # N_sa[(state_key, a)]
        self.Q_sa = {}     # Q_sa[(state_key, a)]
        self.children = {} # children[state_key] = lista de acciones legales

    def act(self, s: np.ndarray) -> int:
        """
        El torneo llama act(state.board): recibimos solo el tablero.
        """
        # Por si el torneo no llamó mount() explícitamente
        if not hasattr(self, "simulations_per_move"):
            self.mount()

        board = self._normalize_board(s)

        # Inferir quién juega: número de fichas 1 y -1
        player_to_move = self._infer_player_to_move(board)

        legal_moves = self._legal_moves(board)
        if not legal_moves:
            # Fallback extremo: jugar cualquier columna no llena
            available_cols = [c for c in range(board.shape[1]) if board[0, c] == 0]
            if not available_cols:
                return 0
            return int(self.rng.choice(available_cols))

        state_key = self._state_key(board, player_to_move)

        # MCTS: simulaciones desde la raíz
        for _ in range(self.simulations_per_move):
            board_copy = board.copy()
            self._search(board_copy, player_to_move)

        # Elegir la acción más visitada en la raíz
        best_move = None
        best_Nsa = -1
        for a in legal_moves:
            key_sa = (state_key, a)
            Nsa = self.N_sa.get(key_sa, 0)
            if Nsa > best_Nsa:
                best_Nsa = Nsa
                best_move = a

        if best_move is None:
            best_move = int(self.rng.choice(legal_moves))

        return int(best_move)

    # -----------------------------------------------------------
    # Núcleo de MCTS
    # -----------------------------------------------------------

    def _search(self, board: np.ndarray, player_to_move: int) -> float:
        """
        Búsqueda recursiva.
        Devuelve el valor desde la perspectiva de 'player_to_move' en este estado.
        """
        state_key = self._state_key(board, player_to_move)

        # Comprobar terminalidad
        winner = self._check_winner(board)
        if winner != 0:
            # +1 si gana el jugador que mueve, -1 si pierde
            return 1.0 if winner == player_to_move else -1.0

        if self._is_full(board):
            return 0.0  # empate

        legal_moves = self._legal_moves(board)
        if not legal_moves:
            return 0.0

        # Nodo hoja: primera vez que lo vemos -> expansión + rollout
        if state_key not in self.N:
            self.N[state_key] = 0
            self.children[state_key] = list(legal_moves)
            for a in legal_moves:
                key_sa = (state_key, a)
                self.N_sa[key_sa] = 0
                self.Q_sa[key_sa] = 0.0
            v = self._rollout(board, player_to_move)
            return v

        # Nodo interno: selección UCT
        N_s = self.N.get(state_key, 0)
        log_N = float(np.log(N_s + 1.0))  # >= 0

        best_score = -np.inf
        best_action = None

        for a in legal_moves:
            key_sa = (state_key, a)
            Nsa = self.N_sa.get(key_sa, 0)
            Qsa = self.Q_sa.get(key_sa, 0.0)

            if Nsa == 0:
                uct = np.inf  # forzar explorar acciones no probadas
            else:
                if log_N > 0.0:
                    exploration = self.c_uct * float(np.sqrt(log_N / float(Nsa)))
                else:
                    exploration = 0.0
                uct = Qsa + exploration

            if uct > best_score:
                best_score = uct
                best_action = a

        if best_action is None:
            best_action = int(self.rng.choice(legal_moves))

        # Transición al siguiente estado
        next_board = self._apply_move(board, best_action, player_to_move)

        # Valor desde la perspectiva del siguiente jugador
        v_child = self._search(next_board, -player_to_move)

        # Suma cero: valor para el jugador actual
        v = -v_child

        # Backup
        self.N[state_key] = self.N.get(state_key, 0) + 1
        key_sa = (state_key, best_action)
        Nsa_old = self.N_sa.get(key_sa, 0)
        Nsa_new = Nsa_old + 1
        self.N_sa[key_sa] = Nsa_new

        Q_old = self.Q_sa.get(key_sa, 0.0)
        self.Q_sa[key_sa] = Q_old + (v - Q_old) / float(Nsa_new)

        return v

    def _rollout(self, board: np.ndarray, starting_player: int) -> float:
        """
        Simulación aleatoria hasta el final.
        Retorna valor desde la perspectiva de 'starting_player'.
        """
        current_board = board.copy()
        current_player = starting_player

        while True:
            winner = self._check_winner(current_board)
            if winner != 0:
                return 1.0 if winner == starting_player else -1.0
            if self._is_full(current_board):
                return 0.0

            legal_moves = self._legal_moves(current_board)
            if not legal_moves:
                return 0.0

            a = int(self.rng.choice(legal_moves))
            current_board = self._apply_move(current_board, a, current_player)
            current_player = -current_player

    # -----------------------------------------------------------
    # Utilidades de estado / juego
    # -----------------------------------------------------------

    def _normalize_board(self, s: np.ndarray) -> np.ndarray:
        """
        Normaliza el tablero a {0, 1, -1}.
        Si hay '2' y no hay '-1', mapea 2 -> -1.
        """
        board = s.astype(np.int8).copy()
        vals = np.unique(board)
        vals_list = vals.tolist()
        if 2 in vals_list and -1 not in vals_list:
            board[board == 2] = -1
        return board

    def _infer_player_to_move(self, board: np.ndarray) -> int:
        """
        Inferencia estándar:
        - Primer movimiento: jugador 1.
        - Luego alternan.
        """
        p1 = int(np.sum(board == 1))
        p2 = int(np.sum(board == -1))
        # Si han puesto el mismo número de fichas -> turno de 1
        # Si 1 tiene una ficha más -> turno de -1
        return 1 if p1 == p2 else -1

    def _state_key(self, board: np.ndarray, player_to_move: int):
        """
        Clave hashable: (bytes del tablero, jugador que mueve).
        """
        return (board.tobytes(), int(player_to_move))

    def _legal_moves(self, board: np.ndarray):
        """
        Columnas donde la fila superior está vacía.
        """
        n_rows, n_cols = board.shape
        return [c for c in range(n_cols) if board[0, c] == 0]

    def _apply_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """
        Deja caer una ficha en la columna 'col' para 'player'.
        """
        new_board = board.copy()
        n_rows, _ = new_board.shape
        for r in range(n_rows - 1, -1, -1):
            if new_board[r, col] == 0:
                new_board[r, col] = player
                return new_board
        # Si la columna está llena, algo fue mal en _legal_moves
        return new_board

    def _is_full(self, board: np.ndarray) -> bool:
        return not np.any(board == 0)

    def _check_winner(self, board: np.ndarray) -> int:
        """
        Devuelve:
        - 1 si gana el jugador 1
        - -1 si gana el jugador -1
        - 0 si no hay ganador
        """
        n_rows, n_cols = board.shape

        # Horizontal
        for r in range(n_rows):
            for c in range(n_cols - 3):
                window = board[r, c:c+4]
                if window[0] != 0 and np.all(window == window[0]):
                    return int(window[0])

        # Vertical
        for c in range(n_cols):
            for r in range(n_rows - 3):
                window = board[r:r+4, c]
                if window[0] != 0 and np.all(window == window[0]):
                    return int(window[0])

        # Diagonal positiva (↗)
        for r in range(3, n_rows):
            for c in range(n_cols - 3):
                window = np.array([board[r - i, c + i] for i in range(4)], dtype=board.dtype)
                if window[0] != 0 and np.all(window == window[0]):
                    return int(window[0])

        # Diagonal negativa (↘)
        for r in range(n_rows - 3):
            for c in range(n_cols - 3):
                window = np.array([board[r + i, c + i] for i in range(4)], dtype=board.dtype)
                if window[0] != 0 and np.all(window == window[0]):
                    return int(window[0])

        return 0
