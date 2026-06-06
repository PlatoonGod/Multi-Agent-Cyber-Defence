"""
DQN Training Script
====================
Trains the DQN defender for 3000 episodes then saves the trained agents.
Saves a checkpoint every 500 episodes — progress survives session cutoffs.
Resumes automatically from the latest checkpoint if one exists.

Run this cell directly in Kaggle (do NOT use subprocess).

Author: Louis Poole — BSc Computer Science & AI, Loughborough University
"""

import os
import pickle
import sys
import gymnasium as gym
import cyberbattle._env.cyberbattle_env  # noqa

from dqn_defender import DQNDefender

WORK_DIR       = os.path.dirname(os.path.abspath(__file__))
NETWORK_SIZE   = 20
TRAIN_EPISODES = 3000
MAX_STEPS      = 200
RESULTS_DIR    = os.path.join(WORK_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
MODEL_PATH     = os.path.join(RESULTS_DIR, "dqn_agents.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    """Print and flush immediately — required for output to appear in notebook cells."""
    print(msg, flush=True)


def unwrap_env(e):
    while hasattr(e, "env"):
        e = e.env
    return e


def find_latest_checkpoint():
    """Return (episode_number, filepath) of the most recent checkpoint, or (0, None)."""
    checkpoints = []
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith("checkpoint_ep") and fname.endswith(".pkl"):
            try:
                ep = int(fname.replace("checkpoint_ep", "").replace(".pkl", ""))
                checkpoints.append((ep, os.path.join(CHECKPOINT_DIR, fname)))
            except ValueError:
                pass
    if not checkpoints:
        return 0, None
    checkpoints.sort(reverse=True)
    return checkpoints[0]


def save_checkpoint(defender, episode_rewards, ep):
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{ep}.pkl")
    payload = {
        "agents": defender.get_agents(),
        "episode_count": defender.episode_count,
        "episode_rewards": episode_rewards,
        "start_episode": ep,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    log(f"  [Checkpoint saved at episode {ep} -> {path}]")


# ---------------------------------------------------------------------------
# Resume or fresh start
# ---------------------------------------------------------------------------

start_ep, checkpoint_path = find_latest_checkpoint()

log("=" * 55)
log("DQN Defender Training")
log(f"Network size  : {NETWORK_SIZE} nodes")
log(f"Total episodes: {TRAIN_EPISODES}")
log(f"Checkpoints   : every 500 episodes -> {CHECKPOINT_DIR}")
log("=" * 55)

defender = DQNDefender(training=True)
env = gym.make("CyberBattleChain-v0", size=NETWORK_SIZE, defender_agent=defender)
base_env = unwrap_env(env)

# Force agent initialisation before loading checkpoint weights
obs, info = env.reset()

episode_rewards = []

if checkpoint_path:
    log(f"\nResuming from checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    for node_id, saved_agent in ckpt["agents"].items():
        if node_id in defender.node_agents:
            defender.node_agents[node_id].policy_net.load_state_dict(
                saved_agent.policy_net.state_dict()
            )
            defender.node_agents[node_id].target_net.load_state_dict(
                saved_agent.target_net.state_dict()
            )
            defender.node_agents[node_id].memory   = saved_agent.memory
            defender.node_agents[node_id].epsilon  = saved_agent.epsilon
    defender.episode_count = ckpt["episode_count"]
    episode_rewards        = ckpt["episode_rewards"]
    start_ep               = ckpt["start_episode"]
    eps_vals = [a.epsilon for a in defender.node_agents.values()]
    log(f"Restored {len(defender.node_agents)} agents from episode {start_ep}.")
    log(f"Current epsilon: {sum(eps_vals)/len(eps_vals):.4f}")
else:
    log("\nNo checkpoint found — starting fresh.")
    start_ep = 0

log("")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

for ep in range(start_ep, TRAIN_EPISODES):
    obs, info = env.reset()
    defender.reset_episode()
    total_reward = 0.0

    for t in range(MAX_STEPS):
        try:
            action = base_env.sample_valid_action()
        except Exception:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    defender.end_episode()
    episode_rewards.append(total_reward)

    # Progress every 100 episodes
    if (ep + 1) % 100 == 0:
        avg      = sum(episode_rewards[-100:]) / 100
        eps_vals = [a.epsilon for a in defender.node_agents.values()]
        mean_eps = sum(eps_vals) / len(eps_vals) if eps_vals else 0
        log(f"  Episode {ep+1:>4}/{TRAIN_EPISODES} | "
            f"Avg Reward (last 100): {avg:>8.2f} | "
            f"Epsilon: {mean_eps:.4f}")

    # Checkpoint every 500 episodes
    if (ep + 1) % 500 == 0:
        save_checkpoint(defender, episode_rewards, ep + 1)

env.close()

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------

trained_agents = defender.get_agents()
with open(MODEL_PATH, "wb") as f:
    pickle.dump(trained_agents, f)

eps_vals = [a.epsilon for a in trained_agents.values()]
log(f"\n{'=' * 55}")
log(f"Training complete.")
log(f"Model saved to : {MODEL_PATH}")
log(f"Episodes trained: {len(episode_rewards)}")
log(f"Final epsilon  : {sum(eps_vals)/len(eps_vals):.4f}")
log(f"{'=' * 55}")
log("Next step: run the experiment cell.")
