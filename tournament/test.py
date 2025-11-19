from typing import Callable
from connect4.dtos import Game, Match, Participant, Versus
from connect4.connect_state import ConnectState
import numpy as np
from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import run_tournament, play

# Read all files within subfolder of "groups"
participants = find_importable_classes("groups", Policy)

# Build a participant list (name, class)
players = list(participants.items())

def play_round(
    versus: Versus,
    play: Callable[[Participant, Participant, int, float, int], Participant],
    best_of: int,
    first_player_distribution: float,
    seed: int,
) -> list[Participant]:
    """Run a round and return the list of winners (handles BYEs)."""
    winners: list[Participant] = []
    for a, b in versus:
        if a is None and b is None:
            raise ValueError("Invalid match: two BYEs")
        if a is None:  # b advances
            winners.append(b)
        elif b is None:  # a advances
            winners.append(a)
        else:
            winners.append(play(a, b, best_of, first_player_distribution, seed))
    return winners

def play(
    a: Participant,
    b: Participant,
    best_of: int,
    first_player_distribution: float,
    seed: int = 911,
) -> Participant:
    """Play a match between two participants and return the winner."""
    # Variables
    a_name, a_policy = a
    b_name, b_policy = b
    a_wins = 0
    b_wins = 0
    draws = 0
    total_games = 0
    games_to_win = (best_of // 2) + 1

    # Random Generator
    rng = np.random.default_rng(seed)

    games: list[Game] = []

    while a_wins < games_to_win and b_wins < games_to_win:
        total_games += 1
        # Decide who goes first based on the distribution
        if rng.random() < first_player_distribution:
            first, second = (a, a_policy()), (b, b_policy())
        else:
            first, second = (b, b_policy()), (a, a_policy())

        # Mount agents
        first[1].mount()
        second[1].mount()

        state = ConnectState()
        state.board = np.array(
    [
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0],
        [1,  0,  0,  0,  0,  0,  0],
        [1,  0,  0,  0,  0,  0,  0],
        [1,  0,  0,  0,  0,  0,  0],
    ],
    dtype=int,
)
        
        game_history: Game = Game()

        while not state.is_final():
            _, current_policy = first if state.player == -1 else second
            action = current_policy.act(state.board)
            game_history.append((state.board.copy().tolist(), int(action)))
            state = state.transition(int(action))
        ConnectState.show(state)
        games.append(game_history)

        # Determine winner
        if state.get_winner() == -1:
            a_wins += 1
        elif state.get_winner() == 1:
            b_wins += 1
        else:
            draws += 1

        # Early stopping in case of too many draws
        if draws >= games_to_win + 5:
            break

    # Save match result
    match = Match(
        player_a=a_name,
        player_b=b_name,
        player_a_wins=a_wins,
        player_b_wins=b_wins,
        draws=draws,
        games=games,
    )

    # Save to file
    match_filename = f"match_{a_name}_vs_{b_name}.json"
    with open("versus/" + match_filename, "w") as f:
        f.write(match.model_dump_json(indent=4))

    if a_wins > 0 or b_wins > 0:
        return a if a_wins > b_wins else b
    # Decide winner at random in case of too many draws with no wins or tie
    return a if rng.random() < 0.5 else b


def next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def make_initial_matches(
    players: list[Participant], shuffle: bool, seed: int
) -> Versus:
    """Create the first round, padding with BYEs (None) up to a power of two."""
    players = players[:]  # copy
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(players)
    size = next_power_of_two(len(players))
    players += [None] * (size - len(players))  # BYEs
    return [(players[i], players[i + 1]) for i in range(0, len(players), 2)]

versus = make_initial_matches(players, shuffle=False, seed=911)
winners = play_round(versus=versus, play=play, best_of=10, first_player_distribution=0.5, seed=911)