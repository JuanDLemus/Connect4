import numpy as np
from tqdm import tqdm

from connect4.connect_state import ConnectState
from groups.PlayerRandom1.policy import Random
from groups.PlayerRules1.policy import RandomBeater1
from groups.PlayerRL1.policy import RLPolicy as RLP1
from groups.PlayerMCTS1.policy import MCTSPolicy as MCTS1
from groups.PlayerRBP1.policy import RandomBeaterPro as RBP1



# --- PARÁMETROS DE EVALUACIÓN ---
NUM_GAMES = 1000  # Número de partidas para evaluar el rendimiento

def evaluate():
    """Función para evaluar el rendimiento del agente RL entrenado."""
    
    print("Iniciando evaluación del agente RL...")
    
    # 1. Inicializar los agentes
    # Cargar nuestro agente RL entrenado en modo de EVALUACIÓN (no entrenamiento)
    # training_mode=False asegura que epsilon sea 0 y solo explote el conocimiento
    rl_agent = RBP1()
    rl_agent.mount() # Carga la Q-Table entrenada


    # El agente oponente que juega al azar
    random_agent = Random()
    random_agent.mount()

    # 2. Contadores de resultados
    rl_agent_wins = 0
    opponent_wins = 0
    draws = 0

    print(f"Evaluando rendimiento en {NUM_GAMES} partidas...")
    
    # Bucle de evaluación
    for game in tqdm(range(NUM_GAMES), desc="Evaluando"):
        state = ConnectState()
        done = False
        
        # Alternar quién empieza para que la evaluación sea justa
        # RL Agent empieza en partidas pares, Random Agent en impares
        rl_turn = -1 if game % 2 == 0 else 1

        while not done:
            current_player = state.player
            
            if current_player == rl_turn:
                # Turno del agente RL
                action = rl_agent.act(state.board)
            else:
                # Turno del agente aleatorio
                action = random_agent.act(state.board)
            
            # Aplicar movimiento
            state = state.transition(action)
            done = state.is_final()

        # 3. Registrar el resultado de la partida
        winner = state.get_winner()
        if winner == rl_turn:
            rl_agent_wins += 1
        elif winner == 0:
            draws += 1
        else:
            opponent_wins += 1
            
    # 4. Calcular y mostrar los resultados finales
    total_games = rl_agent_wins + opponent_wins + draws
    win_rate = (rl_agent_wins / total_games) * 100 if total_games > 0 else 0
    loss_rate = (opponent_wins / total_games) * 100 if total_games > 0 else 0
    draw_rate = (draws / total_games) * 100 if total_games > 0 else 0

    print("\n--- Resultados de la Evaluación ---")
    print(f"Partidas totales: {total_games}")
    print(f"Victorias del Agente RL: {rl_agent_wins}")
    print(f"Victorias del Agente Aleatorio: {opponent_wins}")
    print(f"Empates: {draws}")
    print("-------------------------------------")
    print(f"Tasa de Victorias del Agente RL: {win_rate:.2f}%")
    print(f"Tasa de Derrotas del Agente RL: {loss_rate:.2f}%")
    print(f"Tasa de Empates: {draw_rate:.2f}%")
    print("-------------------------------------")

if __name__ == "__main__":
    evaluate()