# groups/PlayerRL1/pkl_to_dict.py

import gzip
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "q_table.pkl.gz"
OUTPUT_PATH = BASE_DIR / "policy_table_dict.py"


def load_q_table(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def q_to_policy_table(q_table, num_actions: int = 7):
    """
    Convert a heterogeneous Q-table into a policy table.

    Handles:
      - numpy arrays of shape (num_actions,)
      - lists / tuples of length num_actions
      - dicts {action_index: q_value}
    Ignores:
      - scalar floats/ints (likely V(s), useless for policy argmax)
      - unknown types
    """
    policy = {}
    skipped_scalar = 0
    skipped_other = 0

    for state, q_vals in q_table.items():
        best_action = None

        # Case 1: numpy array
        if hasattr(q_vals, "shape"):
            if q_vals.shape[0] == 0:
                continue
            best_action = int(np.argmax(q_vals))

        # Case 2: dict {action: value}
        elif isinstance(q_vals, dict):
            if not q_vals:
                continue
            best_action = max(q_vals, key=q_vals.get)

        # Case 3: list / tuple of Q-values
        elif isinstance(q_vals, (list, tuple)):
            if not q_vals:
                continue
            best_action = max(range(len(q_vals)), key=lambda i: q_vals[i])

        # Case 4: scalar → no action information, skip
        elif isinstance(q_vals, (int, float, np.integer, np.floating)):
            skipped_scalar += 1

        # Anything else → skip and count
        else:
            skipped_other += 1

        if best_action is not None:
            policy[state] = int(best_action)

    print(f"Converted {len(policy)} states to actions.")
    print(f"Skipped {skipped_scalar} scalar-value states and {skipped_other} others.")
    return policy


if __name__ == "__main__":
    print(f"Loading Q-table from {INPUT_PATH} ...")
    q_table = load_q_table(INPUT_PATH)
    print(f"Loaded {len(q_table)} states.")

    policy_table = q_to_policy_table(q_table)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated policy table: state -> best action index\n")
        f.write("POLICY_TABLE = ")
        f.write(repr(policy_table))
        f.write("\n")

    print(f"Policy table written to {OUTPUT_PATH}")
