import numpy as np
import math
import time
import json
import os
from connect4.policy import Policy
# --- Constantes y Configuración ---
C_PARAM = 1.414 
COLUMN_PRIORITIES = [3, 2, 4, 1, 5, 0, 6]

class Node:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        self.untried = [c for c in range(7) if state[0, c] == 0]
        self.wins = 0.0
        self.visits = 0

    def expand(self):
        action = self.untried.pop()
        next_state = apply_move(self.state, action, self.player)
        child_node = Node(next_state, -self.player, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self):
        best = None
        best_val = -float('inf')
        log_n = math.log(self.visits) if self.visits > 0 else 0
        
        for child in self.children:
            if child.visits == 0: return child
            # UCB1
            ucb = (child.wins / child.visits) + C_PARAM * math.sqrt(log_n / child.visits)
            if ucb > best_val:
                best_val = ucb
                best = child
        return best

    def update(self, reward):
        self.visits += 1
        self.wins += reward

# --- Lógica de Tablero ---
def apply_move(state, col, p):
    b = state.copy()
    for r in range(5, -1, -1):
        if b[r, col] == 0:
            b[r, col] = p
            return b
    return b

def check_win(b, p):
    # Verificación rápida
    for r in range(6):
        for c in range(4):
            if b[r,c]==p and b[r,c+1]==p and b[r,c+2]==p and b[r,c+3]==p: return True
    for c in range(7):
        for r in range(3):
            if b[r,c]==p and b[r+1,c]==p and b[r+2,c]==p and b[r+3,c]==p: return True
    for r in range(3):
        for c in range(4):
            if b[r,c]==p and b[r+1,c+1]==p and b[r+2,c+2]==p and b[r+3,c+3]==p: return True
            if b[r+3,c]==p and b[r+2,c+1]==p and b[r+1,c+2]==p and b[r,c+3]==p: return True
    return False

# --- Heurísticas ---
def is_bad_move(state, col, player):
    """Evita regalar victoria inmediata al rival"""
    temp_b = state.copy()
    row = -1
    for r in range(5, -1, -1):
        if temp_b[r, col] == 0:
            temp_b[r, col] = player
            row = r
            break
    if row <= 0: return False # Columna llena o tope
    
    # Check celda superior
    opponent = -player
    temp_b[row - 1, col] = opponent
    if check_win(temp_b, opponent): return True
    return False

def fast_rollout(state, player):
    """Simulación ligera para velocidad"""
    b = state.copy()
    current_p = player
    # Profundidad corta para maximizar simulaciones/segundo
    for _ in range(15):
        valid = [c for c in range(7) if b[0, c] == 0]
        if not valid: return 0 
        move = valid[np.random.randint(len(valid))]
        
        for r in range(5, -1, -1):
            if b[r, move] == 0:
                b[r, move] = current_p
                break
        
        if check_win(b, current_p): return current_p
        current_p = -current_p
    return 0

# --- Motor MCTS ---
def run_mcts(root_state, player, time_limit):
    root = Node(root_state, player)
    start_time = time.time()
    
    while True:
        # Lotes de 50 para evitar overhead de time.time()
        for _ in range(50):
            node = root
            # 1. Selección
            while node.untried == [] and node.children:
                node = node.best_child()
            # 2. Expansión
            if node.untried:
                node = node.expand()
            # 3. Simulación
            winner = fast_rollout(node.state, node.player)
            # 4. Backpropagation
            curr = node
            while curr.parent is not None:
                move_maker = curr.parent.player
                if winner == move_maker: reward = 1.0
                elif winner == -move_maker: reward = 0.0
                else: reward = 0.5
                curr.update(reward)
                curr = curr.parent
            root.visits += 1
            
        if (time.time() - start_time) > time_limit:
            break

    if not root.children:
        valid = [c for c in range(7) if root_state[0, c] == 0]
        return valid[0] if valid else 0
        
    best = max(root.children, key=lambda c: c.visits)
    return best.action

# --- CLASE PRINCIPAL ---

class NewPolicy(Policy):
    def __init__(self):
        self.time_out = 9 # Default
        self.trials_data = {} # Aquí cargaremos el JSON

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.json_file = os.path.join(current_dir, "trials_data.json")

    def mount(self, time_out: int) -> None:
        self.time_out = int(time_out)
        
        # Intentar cargar el JSON subido junto a la política
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    self.trials_data = json.load(f)
                print(f"DEBUG: JSON cargado correctamente desde {self.json_file}")
            except Exception as e:
                print(f"DEBUG: Error leyendo JSON: {e}")
                pass
        else:
            print("DEBUG: No se encontró el archivo JSON adjunto.")
            pass

    def act(self, s: np.ndarray) -> int:
        total = np.count_nonzero(s)
        player = 1 if total % 2 == 0 else -1
        valid_moves = [c for c in range(7) if s[0, c] == 0]
        
        # 1. Instinto (Ganar/Bloquear)
        for m in valid_moves:
            if check_win(apply_move(s, m, player), player): return m
        for m in valid_moves:
            if check_win(apply_move(s, m, -player), -player): return m
            
        # 2. Filtro Taboo
        safe_moves = [m for m in valid_moves if not is_bad_move(s, m, player)]
        if not safe_moves: safe_moves = valid_moves
        if len(safe_moves) == 1: return safe_moves[0]
        
        # 3. MCTS (Limitado a 1.8s para Gradescope)
        limit = min(float(self.time_out) * 0.9, 1.8)
        if total > 30: limit = 0.5 
        
        return run_mcts(s, player, limit)