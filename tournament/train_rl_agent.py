import numpy as np
from tqdm import tqdm

from connect4.connect_state import ConnectState
from groups.PlayerRandom1.policy import Random
from groups.PlayerRules1.policy import RandomBeater1
from groups.PlayerRL1.policy import RLPolicy as RLP1
from groups.PlayerMCTS1.policy import MCTSPolicy as MCTS1

# --- PARÁMETROS DE ENTRENAMIENTO ---
NUM_EPISODES = 10000  # Número de partidas para entrenar
PRINT_EVERY = 1000   # Imprimir progreso cada N partidas
SAVE_EVERY = 500    # Guardar la Q-Table cada N partidas

# --- RECOMPENSAS ---
REWARD_WIN = 20
REWARD_LOSE = -20
REWARD_DRAW = -2
REWARD_STEP = -0.1 # Pequeña penalización por cada movimiento para incentivar victorias rápidas

def train():
    """Función principal para entrenar el agente de Q-Learning."""
    
    # 1. Inicializar los agentes
    # Nuestro agente RL en modo entrenamiento
    rl_agent = RLP1(training_mode=True)
    rl_agent.mount() # Carga la Q-Table si existe, si no, empieza de cero

    # El agente oponente que juega al azar
    random_agent = Random()
    random_agent.mount()

    print("Iniciando entrenamiento del agente RL...")
    print(f"Número de episodios: {NUM_EPISODES}")
    print(f"Agente oponente: {type(random_agent).__name__}")
    
    # Bucle principal de entrenamiento
    for episode in tqdm(range(NUM_EPISODES), desc="Entrenando"):
        # 2. Reiniciar el estado del juego para una nueva partida
        state = ConnectState()
        done = False
        
        # Historial de la partida para el agente RL
        rl_move_history = []

        # Alternar quién empieza
        turn = -1 if episode % 2 == 0 else 1

        while not done:
            current_player = state.player
            
            if current_player == turn: # Turno del agente RL
                # El agente RL elige una acción
                board_state = state.board
                action = rl_agent.act(board_state)

                # Guardamos el estado y la acción para la actualización posterior
                rl_move_history.append({'state': board_state, 'action': action})

                # Realizamos la transición al siguiente estado
                state = state.transition(action)

            else: # Turno del agente aleatorio
                action = random_agent.act(state.board)
                state = state.transition(action)
            
            # Comprobar si la partida ha terminado
            done = state.is_final()

        # 3. La partida ha terminado. Calcular recompensa y actualizar Q-Table.
        winner = state.get_winner()
        
        if not rl_move_history:
            continue

        # Asignar la recompensa final
        final_reward = 0
        if winner == turn:
            final_reward = REWARD_WIN
        elif winner == -turn:
            final_reward = REWARD_LOSE
        else: # Empate
            final_reward = REWARD_DRAW

        # Actualizar la Q-Table para todos los movimientos del agente RL en la partida
        # El último movimiento recibe la recompensa final. Los anteriores, la penalización por paso.
        for i, move in enumerate(reversed(rl_move_history)):
            reward = final_reward if i == 0 else REWARD_STEP
            
            # El "siguiente estado" para un movimiento es el estado antes del siguiente movimiento del RL
            # o el estado final si es el último movimiento.
            if i == 0:
                next_state_board = state.board
            else:
                next_state_board = rl_move_history[-i]['state']
            
            rl_agent.update_q_table(
                state=move['state'],
                action=move['action'],
                reward=reward,
                next_state=next_state_board
            )

        # 4. Decaimiento de Epsilon
        rl_agent.decay_epsilon()

        # 5. Guardado periódico
        if (episode + 1) % SAVE_EVERY == 0:
            rl_agent.save_q_table()
            # print(f"\n[Episodio {episode + 1}] Q-Table guardada. Epsilon actual: {rl_agent.epsilon:.4f}")

    # Guardado final
    rl_agent.save_q_table()
    print("\nEntrenamiento completado. La Q-Table final ha sido guardada.")
    print(f"Epsilon final: {rl_agent.epsilon:.4f}")

if __name__ == "__main__":
    train()