import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState as CS

class RandomBeater1(Policy):
    def mount(self) -> None:
        pass
    def act(self, s: np.ndarray) -> int:

        # Regla 1:
        # Si existe una jugada que deja 4 en línea (para cualquiera de los dos jugadores),
        # se juega ahí: gana si es propia o bloquea si es del rival.
        next_final, cols = self.next_is_final(s)
        if next_final != 0 and cols:
            return int(cols[0])

        # Regla 2:
        # Buscar 3 en línea

        # Regla 3:
        # Busca el centro si está disponible
        n_cols = s.shape[1]
        center = n_cols // 2
        if (CS.is_applicable(center, s)):
            return int(center)
        
        # Regla 4:
        # Juega aleatoriamente en una columna disponible
        rng = np.random.default_rng()
        available_cols = [c for c in range(7) if s[0, c] == 0]
        return int(rng.choice(available_cols))

    def next_is_final(self, s: np.ndarray) -> tuple[int, list[int]]:
        rows, cols = s.shape
        candidate_cols: list[int] = []
        winning_player = 0

        for c in range(cols):
            # Columna no jugable, ya está llena
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
                state = CS(new_board, player)
                if state.get_winner() == player:
                    candidate_cols.append(c)
                    winning_player = player
                    # No hace falta probar el otro jugador en esta columna
                    break

        return winning_player, candidate_cols
