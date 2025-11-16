import numpy as np
import pickle
import os
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import Tuple, Any

class RLPolicy(Policy):
    """
    Una política de agente que utiliza Q-Learning para aprender a jugar Connect4.
    """
    def __init__(self, training_mode: bool = False):
        self.training_mode = training_mode
        # Parámetros de Q-Learning
        self.learning_rate = 0.1  # Alpha: Tasa de aprendizaje
        self.discount_factor = 0.95 # Gamma: Factor de descuento para recompensas futuras
        self.epsilon = 1.0        # Epsilon: Tasa de exploración inicial
        self.epsilon_decay = 0.9995 # Tasa de decaimiento de epsilon
        self.epsilon_min = 0.01   # Tasa de exploración mínima

        # La Q-Table se cargará o creará vacía
        self.q_table: dict[Tuple[Any, int], float] = {}
        self.q_table_path = os.path.join(os.path.dirname(__file__), 'q_table.pkl')

    def _get_state_key(self, board: np.ndarray) -> Tuple:
        """Convierte el tablero (estado) en una clave inmutable para el diccionario Q-Table."""
        return tuple(map(tuple, board))

    def mount(self) -> None:
        """
        Monta el agente. Carga la Q-Table si existe.
        Si no está en modo entrenamiento, desactiva la exploración.
        """
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                self.q_table = pickle.load(f)
        
        if not self.training_mode:
            self.epsilon = 0  # En modo de juego/evaluación, solo explotamos el conocimiento

    def act(self, s: np.ndarray) -> int:
        """
        Decide una acción (columna) a tomar dado un estado del tablero.
        Utiliza una estrategia epsilon-greedy.
        """
        state_key = self._get_state_key(s)
        available_cols = [c for c in range(s.shape[1]) if s[0, c] == 0]

        if not available_cols:
            return 0  # No hay movimientos posibles

        # Decidir entre exploración y explotación
        if self.training_mode and np.random.rand() < self.epsilon:
            # Exploración: elige una acción aleatoria
            return int(np.random.choice(available_cols))
        else:
            # Explotación: elige la mejor acción conocida
            q_values = {col: self.q_table.get((state_key, col), 0) for col in available_cols}
            # Elegir la columna con el valor Q más alto
            max_q_value = -float('inf')
            best_cols = []
            for col, q_value in q_values.items():
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_cols = [col]
                elif q_value == max_q_value:
                    best_cols.append(col)
            
            # Si hay varios mejores, elige uno al azar
            return int(np.random.choice(best_cols))

    def update_q_table(self, state, action, reward, next_state):
        """Actualiza la Q-Table usando la ecuación de Bellman."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        old_value = self.q_table.get((state_key, action), 0)
        
        # Encontrar el valor Q máximo para el siguiente estado
        available_cols = [c for c in range(next_state.shape[1]) if next_state[0, c] == 0]
        if not available_cols:
            next_max = 0 # Estado terminal
        else:
            next_max = max([self.q_table.get((next_state_key, col), 0) for col in available_cols])

        # Ecuación de Q-Learning
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[(state_key, action)] = new_value

    def decay_epsilon(self):
        """Reduce la tasa de exploración (epsilon)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self):
        """Guarda la Q-Table en un archivo."""
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(self.q_table, f)
