"""
Decentralised Multi-Agent Cyber Defender
=========================================
Each LocalDefender observes only its OWN node.
Knowledge of other nodes arrives exclusively via alert messages
in the agent's inbox — this is the core "local-only visibility"
design that makes the communication condition meaningful.

Detection model calibrated to 11-day average attacker dwell time
(Mandiant M-Trends 2025). In 200-step episodes this gives p = 1/11 ≈ 0.09.

Decision model:
  - Suspicion accumulates over time (no immediate detection)
  - Reimage threshold: 0.95
  - Each agent limited to its OWN node only
  - Alerts optionally shared to neighbours

Experimental conditions:
  Condition 3 — DecentralisedDefenderEcosystem(share_alerts=False)
  Condition 4 — DecentralisedDefenderEcosystem(share_alerts=True, neighbour_only=False)
  Condition 5 — DecentralisedDefenderEcosystem(share_alerts=True, neighbour_only=True)

Author: Louis Poole — BSc Computer Science & AI, Loughborough University
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from cyberbattle._env.defender import DefenderAgent, DefenderAgentActions
from cyberbattle.simulation.model import Environment

DETECTION_PROBABILITY = 0.09   # p = 1/11 (11-day dwell time, Mandiant 2025)
SUSPICION_INCREMENT   = 0.40
SUSPICION_DECAY       = 0.05
REIMAGE_THRESHOLD     = 0.95
SCAN_THRESHOLD        = 0.50
MAX_ACTIONS_PER_STEP  = 2


# ---------------------------------------------------------------------------
# Local defender — one instance per node
# ---------------------------------------------------------------------------

@dataclass
class LocalDefender:
    node_id:   str
    inbox:     List[Dict[str, Any]] = field(default_factory=list)
    suspicion: float = field(default=0.0, init=False)

    def observe_own_node(self, environment: Environment) -> float:
        try:
            node_info = environment.get_node(self.node_id)
        except Exception:
            self.suspicion = max(0.0, self.suspicion - SUSPICION_DECAY)
            return self.suspicion

        if node_info.agent_installed:
            if random.random() < DETECTION_PROBABILITY:
                self.suspicion = min(1.0, self.suspicion + SUSPICION_INCREMENT)
            else:
                self.suspicion = max(0.0, self.suspicion - SUSPICION_DECAY)
        else:
            self.suspicion = max(0.0, self.suspicion - SUSPICION_DECAY)

        return self.suspicion

    def neighbor_alert_count(self) -> int:
        return sum(1 for m in self.inbox if m.get("type") == "ALERT")

    def clear_inbox(self):
        self.inbox.clear()

    def decide(self, suspicion: float, neighbor_alerts: int) -> str:
        if suspicion >= REIMAGE_THRESHOLD:
            return "REIMAGE"
        if suspicion >= SCAN_THRESHOLD or neighbor_alerts > 0:
            return "SCAN"
        return "NOOP"


# ---------------------------------------------------------------------------
# Ecosystem
# ---------------------------------------------------------------------------

class DecentralisedDefenderEcosystem(DefenderAgent):

    def __init__(self, node_count: int, share_alerts: bool = True,
                 neighbour_only: bool = False):
        self.node_count     = node_count
        self.share_alerts   = share_alerts
        self.neighbour_only = neighbour_only
        self.locals: List[LocalDefender] = []
        self._initialized   = False
        self._neighbours: Dict[str, List[str]] = {}
        self.actions_taken: Dict[str, int] = {"SCAN": 0, "REIMAGE": 0, "NOOP": 0}

    def _init_locals(self, environment):
        if not self._initialized:
            node_ids = list(environment.network.nodes)
            self.locals = [LocalDefender(nid) for nid in node_ids]
            self.node_count = len(node_ids)
            for nid in node_ids:
                self._neighbours[nid] = list(environment.network.neighbors(nid))
            self._initialized = True

    def _broadcast(self, sender_id, msg: Dict[str, Any]):
        if not self.share_alerts:
            return
        if self.neighbour_only:
            targets = self._neighbours.get(sender_id, [])
            for defender in self.locals:
                if defender.node_id in targets:
                    defender.inbox.append({"from": sender_id, **msg})
        else:
            for defender in self.locals:
                if defender.node_id != sender_id:
                    defender.inbox.append({"from": sender_id, **msg})

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        self._init_locals(environment)

        for defender in self.locals:
            suspicion = defender.observe_own_node(environment)
            alerts    = defender.neighbor_alert_count()
            decision  = defender.decide(suspicion, alerts)
            self.actions_taken[decision] = self.actions_taken.get(decision, 0) + 1

            if decision in ("SCAN", "REIMAGE"):
                self._broadcast(defender.node_id, {
                    "type":      "ALERT",
                    "decision":  decision,
                    "node":      defender.node_id,
                    "suspicion": suspicion,
                    "time":      t,
                })

            defender.clear_inbox()

            if decision == "SCAN":
                if hasattr(actions, "scan_node"):
                    actions.scan_node(defender.node_id)

            elif decision == "REIMAGE":
                if hasattr(actions, "reimage_node"):
                    try:
                        node_info = environment.get_node(defender.node_id)
                        if node_info.reimagable:
                            actions.reimage_node(defender.node_id)
                            defender.suspicion = 0.0
                        else:
                            if hasattr(actions, "scan_node"):
                                actions.scan_node(defender.node_id)
                    except Exception:
                        pass

    def reset_episode(self):
        """Reset only per-episode state. Do NOT reset _initialized —
        the locals list and neighbour map must persist across episodes."""
        self.actions_taken = {"SCAN": 0, "REIMAGE": 0, "NOOP": 0}
        for defender in self.locals:
            defender.inbox.clear()
            defender.suspicion = 0.0
