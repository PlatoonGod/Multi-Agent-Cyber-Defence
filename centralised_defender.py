"""
Centralised Baseline Defender
===============================
A single agent with full global visibility of all nodes.
Used as the comparison baseline in experiments.

Detection model is calibrated to Mandiant's M-Trends 2025 global average
attacker dwell time of 11 days. Mapping 11 days to a 200-step episode gives
a per-step detection probability of 1/11 ≈ 0.09.

Decision model:
  - Suspicion accumulates over time (detection is not immediate)
  - Reimage threshold: 0.95 (decisive action once threat confirmed)
  - Can act on all nodes per timestep (global visibility advantage)
  - Max 2 actions per node per timestep

Experimental condition:
  Condition 2 — CentralisedDefender(node_count=N)

Author: Louis Poole — BSc Computer Science & AI, Loughborough University
"""

import random
from typing import Dict

from cyberbattle._env.defender import DefenderAgent, DefenderAgentActions
from cyberbattle.simulation.model import Environment


# Calibrated to real-world detection time (avg 150 days ~ step 133/200)
DETECTION_PROBABILITY = 0.09  # 11-day dwell time mapped to 200-step episodes   # Per-step chance of spotting attacker
SUSPICION_INCREMENT   = 0.40   # How much suspicion rises per detection event
SUSPICION_DECAY       = 0.05   # Suspicion fades slightly each step if no signal
REIMAGE_THRESHOLD     = 0.95   # Decisive action threshold (slightly raised)
SCAN_THRESHOLD        = 0.50   # Investigate if suspicion elevated
MAX_ACTIONS_PER_NODE  = 2      # Max actions per node per timestep


class CentralisedDefender(DefenderAgent):
    """
    Centralised defender with full global visibility.

    Checks every node every timestep, but detection is probabilistic —
    modelling the reality that attackers often evade immediate detection.
    Suspicion accumulates over time before triggering a response.
    """

    def __init__(self, node_count: int):
        self.node_count = node_count
        self.actions_taken: Dict[str, int] = {"SCAN": 0, "REIMAGE": 0, "NOOP": 0}
        self._node_ids = None
        self._suspicion: Dict[str, float] = {}

    def _get_node_ids(self, environment):
        if self._node_ids is None:
            self._node_ids = list(environment.network.nodes)
            self._suspicion = {nid: 0.0 for nid in self._node_ids}
        return self._node_ids

    def _update_suspicion(self, environment, node_id) -> float:
        """
        Probabilistic detection model.
        If attacker is present, there is a DETECTION_PROBABILITY chance
        of observing a signal this timestep. Suspicion accumulates across
        steps, modelling delayed detection as seen in real-world incidents.
        """
        try:
            node_info = environment.get_node(node_id)
        except Exception:
            return self._suspicion.get(node_id, 0.0)

        current = self._suspicion.get(node_id, 0.0)

        if node_info.agent_installed:
            # Attacker present — may or may not be detected this step
            if random.random() < DETECTION_PROBABILITY:
                current = min(1.0, current + SUSPICION_INCREMENT)
            else:
                # No signal this step — slight decay
                current = max(0.0, current - SUSPICION_DECAY)
        else:
            # Node clean — suspicion decays
            current = max(0.0, current - SUSPICION_DECAY)

        self._suspicion[node_id] = current
        return current

    def _decide(self, suspicion: float) -> str:
        if suspicion >= REIMAGE_THRESHOLD:
            return "REIMAGE"
        if suspicion >= SCAN_THRESHOLD:
            return "SCAN"
        return "NOOP"

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        """
        Called once per timestep. Checks every node globally.
        Can act on all nodes simultaneously (centralised advantage),
        but limited to MAX_ACTIONS_PER_NODE actions per node.
        """
        for node_id in self._get_node_ids(environment):
            suspicion = self._update_suspicion(environment, node_id)
            decision = self._decide(suspicion)
            self.actions_taken[decision] = self.actions_taken.get(decision, 0) + 1

            actions_used = 0

            if decision == "SCAN" and actions_used < MAX_ACTIONS_PER_NODE:
                if hasattr(actions, "scan_node"):
                    actions.scan_node(node_id)
                    actions_used += 1

            elif decision == "REIMAGE" and actions_used < MAX_ACTIONS_PER_NODE:
                if hasattr(actions, "reimage_node"):
                    try:
                        node_info = environment.get_node(node_id)
                        if node_info.reimagable:
                            actions.reimage_node(node_id)
                            self._suspicion[node_id] = 0.0
                        else:
                            if hasattr(actions, "scan_node"):
                                actions.scan_node(node_id)
                        actions_used += 1
                    except Exception:
                        pass

    def reset_episode(self):
        """Reset per-episode state."""
        self.actions_taken = {"SCAN": 0, "REIMAGE": 0, "NOOP": 0}
        self._suspicion = {nid: 0.0 for nid in (self._node_ids or [])}
