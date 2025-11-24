import numpy as np
from tqdm import tqdm

from connect4.connect_state import ConnectState
from groups.PlayerMCTS1.policy import MCTSPolicy as MCTS1
from groups.PlayerRBP1.policy import RandomBeaterPro


# Parámetros de entrenamiento
NUM_EPISODES = 100
SAVE_EVERY = 35
EVAL_EVERY = 50

REWARD_WIN = 20.0
REWARD_LOSE = -20.0
REWARD_DRAW = -4.0
REWARD_STEP = -0.2


def build_opponent():
    mcts_agent = MCTS1()
    mcts_agent.mount()
    return mcts_agent


def evaluate(agent: RandomBeaterPro, opponent, num_games: int = 5):
    """
    Evaluación rápida del agente entrenado contra el oponente MCTS1.
    Se forza a RandomBeaterPro a jugar sin MCTS (solo Q-table + heurísticas)
    para medir la calidad del “destilado” de la política.
    """
    eval_agent = RandomBeaterPro(training_mode=False, mcts_time_limit=0.0, rl_action_prob=1.0)
    eval_agent.mount()
    eval_agent.q_table = dict(agent.q_table)

    wins = draws = loses = 0

    for g in range(num_games):
        state = ConnectState()
        done = False
        turn = -1 if g % 2 == 0 else 1  # alternar quién empieza

        while not done:
            current_player = state.player

            if current_player == turn:
                action = eval_agent.act(state.board)
            else:
                action = opponent.act(state.board)

            state = state.transition(action)
            done = state.is_final()

        winner = state.get_winner()
        if winner == turn:
            wins += 1
        elif winner == -turn:
            loses += 1
        else:
            draws += 1

    total = max(1, wins + draws + loses)
    print(
        f"  vs {type(opponent).__name__}: "
        f"W {wins/total:.3f}  D {draws/total:.3f}  L {loses/total:.3f}"
    )


def schedule_params(agent: RandomBeaterPro, episode_idx: int) -> None:
    """
    Ajusta el uso de MCTS interno y RL según el progreso del entrenamiento:

      - 0–30%: MCTS interno rápido (20 ms), RL ocasional.
      - 30–70%: MCTS interno muy rápido (10 ms), más decisiones RL.
      - 70–100%: sin MCTS interno, casi todo RL (destilado).
    """
    frac = episode_idx / NUM_EPISODES

    if frac < 0.3:
        agent.mcts_time_limit = 0.02   # 20 ms
        agent.rl_action_prob = 0.05    # 5% de jugadas puramente RL
    elif frac < 0.7:
        agent.mcts_time_limit = 0.01   # 10 ms
        agent.rl_action_prob = 0.30    # 30% RL, 70% MCTS interno
    else:
        agent.mcts_time_limit = 0.0    # sin MCTS interno (act() lo apaga)
        agent.rl_action_prob = 0.90    # casi todo RL + heurísticas


def train():
    rl_agent = RandomBeaterPro(training_mode=True, mcts_time_limit=0.02, rl_action_prob=0.05)
    rl_agent.mount()   # carga Q-table si existe

    mcts_opponent = build_opponent()

    print("Entrenando RandomBeaterPro contra MCTS1...")
    print(f"Episodios totales: {NUM_EPISODES}")

    for episode in tqdm(range(NUM_EPISODES), desc="Entrenando"):
        # Ajuste dinámico de MCTS interno / RL según la fase
        schedule_params(rl_agent, episode)

        state = ConnectState()
        done = False
        rl_move_history = []

        # Alternar quién empieza (para evitar sesgo de primer movimiento)
        turn = -1 if episode % 2 == 0 else 1

        while not done:
            current_player = state.player

            if current_player == turn:
                board_state = state.board.copy()
                action = rl_agent.act(board_state)
                rl_move_history.append({"state": board_state, "action": action})
                state = state.transition(action)
            else:
                action = mcts_opponent.act(state.board)
                state = state.transition(action)

            done = state.is_final()

        winner = state.get_winner()

        if not rl_move_history:
            continue

        if winner == turn:
            final_reward = REWARD_WIN
        elif winner == -turn:
            final_reward = REWARD_LOSE
        else:
            final_reward = REWARD_DRAW

        # Último movimiento recibe recompensa final; el resto, penalización por paso
        for i, move in enumerate(reversed(rl_move_history)):
            reward = final_reward if i == 0 else REWARD_STEP

            if i == 0:
                next_state_board = state.board
            else:
                next_state_board = rl_move_history[-i]["state"]

            rl_agent.update_q_table(
                state=move["state"],
                action=move["action"],
                reward=reward,
                next_state=next_state_board,
            )

        rl_agent.decay_epsilon()

        if (episode + 1) % SAVE_EVERY == 0:
            rl_agent.save_q_table()

        if (episode + 1) % EVAL_EVERY == 0:
            print(f"\nEvaluación en episodio {episode + 1}:")
            evaluate(rl_agent, mcts_opponent, num_games=40)
            print("")

    rl_agent.save_q_table()
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    train()
