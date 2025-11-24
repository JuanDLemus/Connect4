"""
metrics_runner.py

Evalúa las políticas de Connect4 y registra métricas detalladas por partida y por agente.
"""

import csv
import time
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parche ligero por si el entorno no tiene typing.override
# ---------------------------------------------------------------------------
import typing as _typing

if not hasattr(_typing, "override"):
    def _override(f):
        return f
    _typing.override = _override  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
from connect4.connect_state import ConnectState
from groups.PlayerRandom1.policy import Random
from groups.PlayerRules1.policy import RandomBeater1
from groups.PlayerRL1.policy import RLPolicy
from groups.PlayerMCTS1.policy import MCTSPolicy as MCTS1
from groups.PlayerRBP1.policy import RandomBeaterPro

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
N_GAMES_PER_PAIR = 100  # puedes bajar este número si se vuelve muy lento
RNG_SEED = 2025

AgentSpec = Tuple[str, type]

AGENTS: List[AgentSpec] = [
    ("Random", Random),
    ("RandomBeater1", RandomBeater1),
    ("RLPolicy", RLPolicy),
    ("MCTS1", MCTS1),
    ("RandomBeaterPro", RandomBeaterPro),
]


@dataclass
class GameMetricsRow:
    # Identificación
    game_id: int
    pair_id: str          # "AgentA_vs_AgentB"
    agent: str
    opponent: str
    agent_is_first: int   # 1 si juega como primer jugador (-1), 0 si juega como segundo

    # Resultado
    result: int           # +1 victoria, 0 tablas, -1 derrota
    result_label: str     # "win" / "draw" / "loss"

    # Métricas básicas
    moves_agent: int               # número de movimientos hechos por este agente
    moves_total: int               # longitud total de la partida
    complexity_total_ns: int       # suma de tiempos de decisión en ns (proxy de complejidad computacional)
    response_time_mean_ms: float   # tiempo medio por decisión (ms) en esta partida
    response_time_min_ms: float    # mínimo tiempo por decisión (ms)
    response_time_max_ms: float    # máximo tiempo por decisión (ms)

    # Información del espacio de acción
    available_actions_mean: float
    available_actions_min: int
    available_actions_max: int

    # Reproducibilidad
    random_seed: int


# ---------------------------------------------------------------------------
# Motor de partida instrumentado
# ---------------------------------------------------------------------------

def play_instrumented_game(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    game_id: int,
    rng: np.random.Generator,
) -> Tuple[GameMetricsRow, GameMetricsRow]:
    """
    Juega UNA partida entre agent_a y agent_b, eligiendo aleatoriamente quién empieza
    y devolviendo una fila de métricas por agente.
    """
    name_a, cls_a = agent_a
    name_b, cls_b = agent_b
    pair_id = f"{name_a}_vs_{name_b}"

    # 1 si A empieza (-1), 0 si B empieza
    a_starts = bool(rng.integers(0, 2))

    if a_starts:
        first_name, first_cls = name_a, cls_a
        second_name, second_cls = name_b, cls_b
    else:
        first_name, first_cls = name_b, cls_b
        second_name, second_cls = name_a, cls_a

    # Instanciar políticas
    first_policy = first_cls()
    second_policy = second_cls()
    first_policy.mount()
    second_policy.mount()

    metrics: Dict[str, Dict[str, object]] = {
        first_name: {
            "seat": "first",
            "decision_times_ns": [],     # List[int]
            "available_actions": [],     # List[int]
            "moves": 0,
        },
        second_name: {
            "seat": "second",
            "decision_times_ns": [],
            "available_actions": [],
            "moves": 0,
        },
    }

    state = ConnectState()
    moves_total = 0
    game_seed = int(rng.integers(0, 2**63 - 1))

    # Bucle de juego
    while not state.is_final():
        current_player = state.player  # -1 (primero) o 1 (segundo)
        if current_player == -1:
            current_name = first_name
            current_policy = first_policy
        else:
            current_name = second_name
            current_policy = second_policy

        # Acciones legales disponibles
        available_cols = int(np.count_nonzero(state.board[0, :] == 0))
        metrics[current_name]["available_actions"].append(available_cols)

        # Medición del tiempo de respuesta
        t0 = time.perf_counter_ns()
        action = current_policy.act(state.board)
        t1 = time.perf_counter_ns()
        dt_ns = t1 - t0

        metrics[current_name]["decision_times_ns"].append(dt_ns)
        metrics[current_name]["moves"] += 1

        # Transición de estado
        state = state.transition(int(action))
        moves_total += 1

        # Tope de seguridad (Connect4 máximo 42 movimientos)
        if moves_total > 42:
            break

    winner = state.get_winner()  # -1 gana primero, +1 gana segundo, 0 tablas

    rows: Dict[str, GameMetricsRow] = {}

    for agent_name, opponent_name in [(name_a, name_b), (name_b, name_a)]:
        m = metrics[agent_name]
        decision_times_ns: List[int] = m["decision_times_ns"]  # type: ignore[assignment]
        available_actions: List[int] = m["available_actions"]  # type: ignore[assignment]
        moves_agent: int = m["moves"]  # type: ignore[assignment]
        seat: str = m["seat"]  # type: ignore[assignment]

        complexity_total_ns = int(sum(decision_times_ns))

        if decision_times_ns:
            response_time_mean_ms = (sum(decision_times_ns) / len(decision_times_ns)) / 1e6
            response_time_min_ms = min(decision_times_ns) / 1e6
            response_time_max_ms = max(decision_times_ns) / 1e6
        else:
            response_time_mean_ms = 0.0
            response_time_min_ms = 0.0
            response_time_max_ms = 0.0

        if available_actions:
            available_actions_mean = float(sum(available_actions) / len(available_actions))
            available_actions_min = int(min(available_actions))
            available_actions_max = int(max(available_actions))
        else:
            available_actions_mean = 0.0
            available_actions_min = 0
            available_actions_max = 0

        # Resultado desde la perspectiva de este agente
        if winner == 0:
            result = 0
            result_label = "draw"
        else:
            if (winner == -1 and seat == "first") or (winner == 1 and seat == "second"):
                result = 1
                result_label = "win"
            else:
                result = -1
                result_label = "loss"

        agent_is_first = 1 if seat == "first" else 0

        row = GameMetricsRow(
            game_id=game_id,
            pair_id=pair_id,
            agent=agent_name,
            opponent=opponent_name,
            agent_is_first=agent_is_first,
            result=result,
            result_label=result_label,
            moves_agent=moves_agent,
            moves_total=moves_total,
            complexity_total_ns=complexity_total_ns,
            response_time_mean_ms=response_time_mean_ms,
            response_time_min_ms=response_time_min_ms,
            response_time_max_ms=response_time_max_ms,
            available_actions_mean=available_actions_mean,
            available_actions_min=available_actions_min,
            available_actions_max=available_actions_max,
            random_seed=game_seed,
        )
        rows[agent_name] = row

    return rows[name_a], rows[name_b]


# ---------------------------------------------------------------------------
# Bucle principal: barrer todas las parejas de agentes
# ---------------------------------------------------------------------------

def run_experiments() -> None:
    rng = np.random.default_rng(RNG_SEED)
    records: List[GameMetricsRow] = []

    game_counter = 0
    pairs = list(itertools.combinations(AGENTS, 2))

    for (agent_a, agent_b) in pairs:
        name_a, _ = agent_a
        name_b, _ = agent_b
        pair_id = f"{name_a}_vs_{name_b}"
        print(f"\n=== Evaluando pareja: {pair_id} ===")

        for i in range(N_GAMES_PER_PAIR):
            game_counter += 1
            row_a, row_b = play_instrumented_game(agent_a, agent_b, game_counter, rng)
            records.append(row_a)
            records.append(row_b)

            if (i + 1) % 100 == 0:
                print(f"  - Partidas jugadas para {pair_id}: {i + 1}/{N_GAMES_PER_PAIR}")

    if not records:
        print("No se generaron registros.")
        return

    fieldnames = list(asdict(records[0]).keys())
    csv_path = "metrics_raw.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    print(
        f"\nSe guardaron {len(records)} filas en {csv_path} "
        f"({len(records) // 2} partidas en total)."
    )


if __name__ == "__main__":
    run_experiments()
