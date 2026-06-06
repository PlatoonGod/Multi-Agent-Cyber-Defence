"""
DQN Defender
=============
A Deep Q-Network based defender agent that learns a defensive policy
through reinforcement learning. Each node has its own DQN agent with
local-only observation — matching the decentralised condition for
fair comparison.

Training: 3000 episodes
Testing:  500 episodes (frozen policy)

Author: Louis Poole — BSc Computer Science & AI, Loughborough University
"""

import random
import numpy as np
from collections import deque
from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.optim as optim

from cyberbattle._env.defender import DefenderAgent, DefenderAgentActions
from cyberbattle.simulation.model import Environment

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DETECTION_PROBABILITY = 0.09  # 11-day dwell time
SUSPICION_INCREMENT   = 0.40
SUSPICION_DECAY       = 0.05

STATE_SIZE    = 4    # [agent_installed_signal, suspicion, inbox_alerts, timestep_norm]
ACTION_SIZE   = 3    # 0=NOOP, 1=SCAN, 2=REIMAGE

LEARNING_RATE = 0.001
GAMMA         = 0.95   # Discount factor
EPSILON_START = 1.0    # Exploration start
EPSILON_MIN   = 0.05   # Minimum exploration
EPSILON_DECAY = 0.9990  # Reaches ~0.05 after 3000 episodes  # Per-episode decay
BATCH_SIZE    = 64
MEMORY_SIZE   = 10000
TARGET_UPDATE = 10     # Update target network every N episodes


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Per-node DQN Agent
# ---------------------------------------------------------------------------

class NodeDQNAgent:
    """
    A DQN agent responsible for defending a single node.
    Maintains its own replay buffer, network, and suspicion score.
    """

    def __init__(self, node_id: str, training: bool = True):
        self.node_id    = node_id
        self.training   = training
        self.suspicion  = 0.0
        self.inbox: List[Dict[str, Any]] = []

        self.epsilon = EPSILON_START if training else EPSILON_MIN

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.policy_net = DQNNetwork(STATE_SIZE, ACTION_SIZE)
        self.target_net = DQNNetwork(STATE_SIZE, ACTION_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.last_state  = None
        self.last_action = None

    def get_state(self, environment: Environment, t: int) -> np.ndarray:
        """Build local state vector from own-node observation only."""
        try:
            node_info = environment.get_node(self.node_id)
            if node_info.agent_installed:
                if random.random() < DETECTION_PROBABILITY:
                    self.suspicion = min(1.0, self.suspicion + SUSPICION_INCREMENT)
                else:
                    self.suspicion = max(0.0, self.suspicion - SUSPICION_DECAY)
                detected = 1.0
            else:
                self.suspicion = max(0.0, self.suspicion - SUSPICION_DECAY)
                detected = 0.0
        except Exception:
            detected = 0.0
            self.suspicion = max(0.0, self.suspicion - SUSPICION_DECAY)

        inbox_alerts = min(len(self.inbox), 10) / 10.0
        timestep_norm = min(t, 200) / 200.0

        return np.array([detected, self.suspicion, inbox_alerts, timestep_norm],
                       dtype=np.float32)

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if self.training and random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            tensor = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.policy_net(tensor)
            return q_vals.argmax().item()

    def compute_reward(self, environment: Environment, action: int) -> float:
        """Defender-side reward signal."""
        try:
            node_info = environment.get_node(self.node_id)
            compromised = node_info.agent_installed
        except Exception:
            compromised = False

        if action == 2:  # REIMAGE
            if compromised:
                return 10.0   # Successfully cleared attacker
            else:
                return -5.0   # Unnecessary reimage — heavy penalty to prevent reward hacking
        elif action == 1:  # SCAN
            if compromised:
                return 1.0    # Useful scan
            else:
                return -0.1   # Mild penalty for unnecessary scan
        else:  # NOOP
            if compromised:
                return -2.0   # Attacker present but ignored
            else:
                return 0.5    # Node clean, modest reward

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Sample from replay buffer and update policy network."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def reset_episode(self):
        self.suspicion = 0.0
        self.inbox.clear()
        self.last_state  = None
        self.last_action = None

    def clear_inbox(self):
        self.inbox.clear()


# ---------------------------------------------------------------------------
# DQN Defender Ecosystem
# ---------------------------------------------------------------------------

class DQNDefender(DefenderAgent):
    """
    Decentralised DQN defender — one neural network agent per node.
    Each agent observes only its own node (local observation),
    matching the decentralised heuristic conditions for fair comparison.

    Mode:
      training=True  — agents explore and learn (epsilon-greedy)
      training=False — agents exploit learned policy (frozen weights)
    """

    def __init__(self, training: bool = True, shared_agents=None):
        self.training      = training
        self.shared_agents = shared_agents  # Pre-trained agents passed in for test phase
        self.node_agents: Dict[str, NodeDQNAgent] = {}
        self._initialized  = False
        self.episode_count = 0
        self.actions_taken: Dict[str, int] = {"SCAN": 0, "REIMAGE": 0, "NOOP": 0}

    def _init_agents(self, environment):
        if not self._initialized:
            node_ids = list(environment.network.nodes)
            if self.shared_agents:
                self.node_agents = self.shared_agents
            else:
                self.node_agents = {
                    nid: NodeDQNAgent(nid, training=self.training)
                    for nid in node_ids
                }
            self._initialized = True

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        self._init_agents(environment)

        for node_id, agent in self.node_agents.items():
            state = agent.get_state(environment, t)

            # Store experience from last step
            if agent.last_state is not None and self.training:
                reward = agent.compute_reward(environment, agent.last_action)
                agent.store_experience(
                    agent.last_state, agent.last_action,
                    reward, state, False
                )
                agent.train_step()

            action = agent.select_action(state)
            agent.last_state  = state
            agent.last_action = action

            action_name = ["NOOP", "SCAN", "REIMAGE"][action]
            self.actions_taken[action_name] = self.actions_taken.get(action_name, 0) + 1

            agent.clear_inbox()

            if action == 1:  # SCAN
                if hasattr(actions, "scan_node"):
                    actions.scan_node(node_id)

            elif action == 2:  # REIMAGE
                if hasattr(actions, "reimage_node"):
                    try:
                        node_info = environment.get_node(node_id)
                        if node_info.reimagable:
                            actions.reimage_node(node_id)
                            agent.suspicion = 0.0
                        else:
                            if hasattr(actions, "scan_node"):
                                actions.scan_node(node_id)
                    except Exception:
                        pass

    def end_episode(self):
        """Call at end of each training episode to update targets and decay epsilon."""
        if not self.training:
            return
        self.episode_count += 1
        for agent in self.node_agents.values():
            agent.decay_epsilon()
            if self.episode_count % TARGET_UPDATE == 0:
                agent.update_target()

    def reset_episode(self):
        self.actions_taken = {"SCAN": 0, "REIMAGE": 0, "NOOP": 0}
        # Do NOT reset _initialized — node_agents persist across episodes
        for agent in self.node_agents.values():
            agent.reset_episode()

    def get_agents(self) -> Dict[str, NodeDQNAgent]:
        """Return trained agents for use in test phase."""
        return self.node_agents
