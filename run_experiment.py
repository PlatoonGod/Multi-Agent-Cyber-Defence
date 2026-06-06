"""
Experiment Runner
==================
Runs all 6 experimental conditions across 5 independent runs.
Saves per-run CSVs and a combined master CSV.

Conditions:
  1. no_defender                — attacker runs unopposed
  2. centralised                — single agent, full global visibility
  3. decentralised_nocomms      — local agents, no alert sharing
  4. decentralised_cooperative  — local agents, full broadcast alerts
  5. decentralised_competitive  — local agents, neighbour-only alerts
  6. rl_defender                — trained DQN, frozen policy

Output (in results/):
  run_1_results.csv ... run_5_results.csv  — raw per-episode data per run
  all_runs_results.csv                     — combined master dataset
  run_1_summary.csv  ... run_5_summary.csv — mean +/- std per condition per run

Author: Louis Poole — BSc Computer Science & AI, Loughborough University
"""

import csv
import os
import pickle
import statistics
import gymnasium as gym
import cyberbattle._env.cyberbattle_env  # noqa

from centralised_defender import CentralisedDefender
from decentralised_defender import DecentralisedDefenderEcosystem
from dqn_defender import DQNDefender

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORK_DIR     = os.path.dirname(os.path.abspath(__file__))
NETWORK_SIZE = 20
NUM_EPISODES = 500
MAX_STEPS    = 200
NUM_RUNS     = 5
RESULTS_DIR  = os.path.join(WORK_DIR, "results")
DQN_PATH     = os.path.join(RESULTS_DIR, "dqn_agents.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Load trained DQN agents
# ---------------------------------------------------------------------------

try:
    with open(DQN_PATH, "rb") as f:
        TRAINED_DQN_AGENTS = pickle.load(f)
    log(f"Loaded trained DQN agents from {DQN_PATH}")
except FileNotFoundError:
    TRAINED_DQN_AGENTS = None
    log("WARNING: No trained DQN model found. Run the training cell first.")
    log(f"Expected path: {DQN_PATH}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unwrap_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def make_env(defender_agent=None):
    return gym.make(
        "CyberBattleChain-v0",
        size=NETWORK_SIZE,
        defender_agent=defender_agent,
    )


def count_compromised(environment) -> int:
    count = 0
    for node_id in environment.network.nodes:
        try:
            if environment.get_node(node_id).agent_installed:
                count += 1
        except Exception:
            pass
    return count


def run_episode(env, base_env, defender) -> dict:
    """Run one episode and return metrics dict."""
    obs, info = env.reset()

    if defender is not None and hasattr(defender, "reset_episode"):
        defender.reset_episode()

    total_reward             = 0.0
    steps                    = 0
    terminated_early         = False
    time_to_first_compromise = None

    for t in range(MAX_STEPS):
        try:
            action = base_env.sample_valid_action()
        except Exception:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1

        if time_to_first_compromise is None:
            try:
                n = count_compromised(base_env._CyberBattleEnv__environment)
                if n > 1:
                    time_to_first_compromise = t
            except Exception:
                pass

        if terminated or truncated:
            if terminated:
                terminated_early = True
            break

    compromised = 0
    try:
        compromised = count_compromised(base_env._CyberBattleEnv__environment)
    except Exception:
        pass

    return {
        "total_reward":             round(total_reward, 4),
        "steps":                    steps,
        "compromised_nodes":        compromised,
        "terminated_early":         int(terminated_early),
        "time_to_first_compromise": time_to_first_compromise if time_to_first_compromise is not None else MAX_STEPS,
    }


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

def get_conditions():
    return [
        {
            "name":  "no_defender",
            "label": "No Defender",
            "defender_fn": lambda: None,
        },
        {
            "name":  "centralised",
            "label": "Centralised Defender",
            "defender_fn": lambda: CentralisedDefender(node_count=NETWORK_SIZE),
        },
        {
            "name":  "decentralised_nocomms",
            "label": "Decentralised (no alerts)",
            "defender_fn": lambda: DecentralisedDefenderEcosystem(
                node_count=NETWORK_SIZE, share_alerts=False
            ),
        },
        {
            "name":  "decentralised_cooperative",
            "label": "Decentralised (cooperative)",
            "defender_fn": lambda: DecentralisedDefenderEcosystem(
                node_count=NETWORK_SIZE, share_alerts=True, neighbour_only=False
            ),
        },
        {
            "name":  "decentralised_competitive",
            "label": "Decentralised (competitive)",
            "defender_fn": lambda: DecentralisedDefenderEcosystem(
                node_count=NETWORK_SIZE, share_alerts=True, neighbour_only=True
            ),
        },
        {
            "name":  "rl_defender",
            "label": "RL Defender (DQN)",
            "defender_fn": lambda: DQNDefender(
                training=False, shared_agents=TRAINED_DQN_AGENTS
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(run_number: int):
    """Run all 6 conditions for NUM_EPISODES episodes. Returns path to results CSV."""
    results_path = os.path.join(RESULTS_DIR, f"run_{run_number}_results.csv")
    summary_path = os.path.join(RESULTS_DIR, f"run_{run_number}_summary.csv")

    fieldnames = [
        "run", "condition", "episode",
        "total_reward", "steps", "compromised_nodes",
        "terminated_early", "time_to_first_compromise",
    ]

    summary_data = []

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for condition in get_conditions():
            log(f"\n  {'─'*45}")
            log(f"  Condition: {condition['label']}  ({NUM_EPISODES} episodes)")
            log(f"  {'─'*45}")

            rewards     = []
            compromised = []
            steps_list  = []
            ttfc_list   = []

            # Create env and defender ONCE per condition — not once per episode
            defender = condition["defender_fn"]()
            env      = make_env(defender_agent=defender)
            base_env = unwrap_env(env)

            for ep in range(NUM_EPISODES):
                metrics = run_episode(env, base_env, defender)

                writer.writerow({
                    "run":       run_number,
                    "condition": condition["name"],
                    "episode":   ep,
                    **metrics,
                })
                csvfile.flush()

                rewards.append(metrics["total_reward"])
                compromised.append(metrics["compromised_nodes"])
                steps_list.append(metrics["steps"])
                ttfc_list.append(metrics["time_to_first_compromise"])

                if (ep + 1) % 50 == 0:
                    action_str = ""
                    if defender is not None and hasattr(defender, "actions_taken"):
                        a = defender.actions_taken
                        action_str = (f" | SCAN {a.get('SCAN',0):>4} "
                                      f"REIMAGE {a.get('REIMAGE',0):>4} "
                                      f"NOOP {a.get('NOOP',0):>4}")
                    log(f"    Ep {ep+1:>3}/{NUM_EPISODES} | "
                        f"Reward {metrics['total_reward']:>8.2f} | "
                        f"Compromised {metrics['compromised_nodes']}"
                        f"{action_str}")

            env.close()

            def mean(lst): return sum(lst) / len(lst) if lst else 0.0
            def std(lst):  return statistics.stdev(lst) if len(lst) > 1 else 0.0

            summary_data.append({
                "run":              run_number,
                "condition":        condition["name"],
                "label":            condition["label"],
                "mean_reward":      round(mean(rewards),     4),
                "std_reward":       round(std(rewards),      4),
                "mean_compromised": round(mean(compromised), 4),
                "std_compromised":  round(std(compromised),  4),
                "mean_steps":       round(mean(steps_list),  2),
                "std_steps":        round(std(steps_list),   2),
                "mean_ttfc":        round(mean(ttfc_list),   2),
                "std_ttfc":         round(std(ttfc_list),    2),
            })

            s = summary_data[-1]
            log(f"\n  Result: mean compromised = {s['mean_compromised']:.3f} "
                f"+/- {s['std_compromised']:.3f} | "
                f"mean reward = {s['mean_reward']:.2f} "
                f"+/- {s['std_reward']:.2f}")

    # Write per-run summary
    summary_fields = [
        "run", "condition", "label",
        "mean_reward", "std_reward",
        "mean_compromised", "std_compromised",
        "mean_steps", "std_steps",
        "mean_ttfc", "std_ttfc",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_data)

    log(f"\n  Run {run_number} saved: {results_path}")
    return results_path


# ---------------------------------------------------------------------------
# Main — 5 independent runs
# ---------------------------------------------------------------------------

log("=" * 55)
log("Experiment Runner")
log(f"Network size : {NETWORK_SIZE} nodes")
log(f"Episodes     : {NUM_EPISODES} per condition")
log(f"Runs         : {NUM_RUNS}")
log(f"Conditions   : 6")
log(f"Total episodes: {NUM_EPISODES * 6 * NUM_RUNS:,}")
log("=" * 55)

if TRAINED_DQN_AGENTS is None:
    log("\nERROR: Cannot run experiment without trained DQN agents.")
    log(f"Run the training cell first. Model should be at: {DQN_PATH}")
    raise SystemExit(1)

run_paths = []
for run in range(1, NUM_RUNS + 1):
    log(f"\n{'='*55}")
    log(f"RUN {run} of {NUM_RUNS}")
    log(f"{'='*55}")
    run_paths.append(run_once(run))

# Combine all runs into master CSV
master_path = os.path.join(RESULTS_DIR, "all_runs_results.csv")
with open(master_path, "w", newline="") as outfile:
    writer = None
    for path in run_paths:
        with open(path, "r") as infile:
            reader = csv.DictReader(infile)
            if writer is None:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
            writer.writerows(reader)

log(f"\n{'='*55}")
log(f"All runs complete.")
log(f"Master dataset : {master_path}")
log(f"{'='*55}")
log("Next step: run the analysis cell.")
