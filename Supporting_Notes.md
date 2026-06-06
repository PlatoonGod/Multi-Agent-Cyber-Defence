# Project Progress Notes
## Decentralised Multi-Agent Cyber Defence Simulation

**Author:** Louis Poole
**Programme:** BSc Computer Science & Artificial Intelligence
**University:** Loughborough University
**Supervisor:** Dr Andrea Soltoggio
**Document purpose:** Running record of progress, discoveries, decisions and plans for use in thesis write-up

---

## Current Status

The project has completed Phases 1–4. All heuristic and DQN defender implementations are complete. Multiple training and evaluation runs have been conducted across Google Colab and Kaggle. The final experimental run is currently in progress on Kaggle Notebooks using updated parameters (20-node network, 11-day dwell time calibration, 5 independent runs × 500 episodes). DQN training (3000 episodes) is running as of the latest session.

---

## Phase 1–2: Environment Setup & Validation

### Environment
- Simulator: **CyberBattleSim** (Microsoft Research), integrated with **Gymnasium**
- OS: WSL2 Ubuntu 24.04 on Windows (local development)
- Python: 3.12 (Kaggle/Colab runtime)
- Network: **CyberBattleChain-v0**, size=20 nodes (must be even; scaled up from initial 12)

### Dependencies
```
gymnasium==0.29.1, networkx, matplotlib, pyyaml, boolean.py, torch,
scipy, numpy<2.0 (critical — numpy 2.x breaks CyberBattleSim)
```

### Key environment discovery
Node IDs in CyberBattleSim are **strings** (e.g. `'start'`, `'1_LinuxNode'`), not integers. Iterating with `range(node_count)` silently fails — defenders must call `list(environment.network.nodes)` at runtime to get actual IDs. This was a critical silent bug that caused defenders to do nothing for entire episodes.

---

## Phase 3: Heuristic Defender Architecture

### Architecture
```
Simulator
    |
Central Defender Interface (DefenderAgent)
    |
DecentralisedDefenderEcosystem / CentralisedDefender
    |
LocalDefender[node_0..node_N]
```

### Key Design Decisions

#### 1. Detection Probability Model — Final Calibration
Detection probability was updated from 0.15 to **0.09**, recalibrated to an **11-day attacker dwell time** (revised from the original 100–200 day figure). The mapping logic is:

- Episode length = 200 steps = full attack lifecycle
- Mean detection day = 11 → detection step = (11/200) × 200 = step 11
- Per-step detection probability to first detect at step 11: 1/11 ≈ **0.09**

This models a more aggressive detection environment where defenders identify intrusions much earlier, reflecting improvements in modern threat detection tooling.

**Previous value:** 0.15 (100–200 day dwell time)
**Final value:** 0.09 (11-day dwell time)

#### 2. Suspicion Accumulation Model — Updated
Suspicion parameters were updated to improve differentiation between cooperative and competitive alert sharing conditions:

- `+0.40` suspicion per detected signal (increased from +0.30)
- `-0.05` decay per step with no signal (increased from -0.02)

The faster decay means stale alerts matter less and the timing of alert propagation becomes more consequential — which should better differentiate full broadcast (cooperative) from neighbour-only (competitive) sharing.

**Previous:** +0.30 increment, -0.02 decay
**Final:** +0.40 increment, -0.05 decay

#### 3. Reimage Threshold: 0.95
Unchanged. Reflects decisive action with strong accumulated evidence, without requiring absolute certainty.

#### 4. Action Constraints (Fairness Model)
- **Centralised defender:** can act on all nodes per timestep, max 2 actions per node per step
- **Decentralised defenders:** each agent limited to its own node only, max 2 actions per step

#### 5. Cooperative vs Competitive Alert Sharing

| Model | Behaviour | Real-World Analogy |
|-------|-----------|-------------------|
| Cooperative | Full network broadcast | Internal SOC team sharing a common platform |
| Competitive | Neighbours-only broadcast | Inter-company sharing via ISAC |

---

## Experimental Conditions

| # | Condition | Class |
|---|-----------|-------|
| 1 | No Defender | None |
| 2 | Centralised | `CentralisedDefender` |
| 3 | Decentralised (no alerts) | `DecentralisedDefenderEcosystem(share_alerts=False)` |
| 4 | Decentralised (cooperative) | `DecentralisedDefenderEcosystem(share_alerts=True, neighbour_only=False)` |
| 5 | Decentralised (competitive) | `DecentralisedDefenderEcosystem(share_alerts=True, neighbour_only=True)` |
| 6 | RL Defender (DQN) | `DQNDefender(training=False, shared_agents=TRAINED_AGENTS)` |

---

## Preliminary Results (500 Episodes, 12-Node Network, Original Parameters)

These results are from earlier runs and are superseded by the final dataset currently being collected.

### Summary Table

| Condition | Mean Compromised | Mean Reward |
|-----------|-----------------|-------------|
| No Defender | 2.74 ± 0.67 | 242.63 |
| Centralised | 1.35 ± 0.55 | 225.24 |
| Decentralised (no alerts) | 1.39 ± 0.56 | 232.07 |
| Decentralised (cooperative) | 1.37 ± 0.53 | 231.69 |
| Decentralised (competitive) | 1.37 ± 0.55 | 230.95 |

### Statistical Significance (Mann-Whitney U, primary test)
All conditions significantly reduced compromised nodes vs no defender (p ≈ 0.000). Heuristic conditions were statistically indistinguishable from each other (p > 0.26). Cooperative vs competitive: p = 0.534 (ns).

---

## Key Findings from Preliminary Runs

### Finding 1: All defenders significantly reduce compromised nodes
Every defender condition reduces compromised nodes from ~2.74 to ~1.35–1.41 (p≈0.000). Answers **RQ1**.

### Finding 2: Decentralised (no alerts) nearly matches centralised
Despite no global visibility, decentralised local-only agents match centralised performance. Answers **RQ3**.

### Finding 3: Cooperative vs competitive — no significant difference
Statistically indistinguishable on chain topology. Policy implication: ISAC-style neighbour-only sharing is defensively sufficient. The 20-node network with updated parameters is intended to test whether this holds at greater scale and with faster-decaying suspicion.

### Finding 4: DQN outperforms heuristics (with corrected reward function)
After fixing reward hacking (see bugs log), the DQN reduced mean attacker reward to ~49 and mean compromised nodes to ~1.01 in the 12-node run — substantially better than all heuristic conditions.

---

## Methodology Evolution & Key Decisions Log

### Decision: Scale from 12 to 20 nodes (final)
12 nodes was sufficient for initial results but the chain topology meant every node was within 1–2 hops of any other, limiting the propagation advantage of full broadcast over neighbour-only. 20 nodes increases the maximum hop distance and gives alert cascades more room to differentiate. Must remain even.

### Decision: Updated detection probability to 0.09 (11-day dwell time)
The original 0.15 figure was based on 100–200 day dwell times. Updated to reflect the 11-day figure from more recent threat intelligence data, producing a more aggressive detection environment.

### Decision: Faster suspicion decay (0.05)
Increased from 0.02 to make stale alerts matter less. When suspicion decays quickly, the timing of alert propagation becomes more important — full broadcast (cooperative) should benefit more from this than neighbour-only (competitive), potentially revealing a meaningful difference at 20 nodes.

### Decision: 5 independent runs × 500 episodes
Moved from a single run to 5 independent runs to improve statistical validity. Each run produces 500 episodes per condition; across 5 runs this gives 2,500 episodes per condition, tightening variance estimates and enabling inter-run consistency checks. Results are stored per-run and combined into a master CSV.

### Decision: Migrate from Google Colab to Kaggle Notebooks
Colab sessions timed out repeatedly during long training runs. Kaggle offers 30 hours/week of free compute with no idle timeout — sessions persist while a cell is actively running. Setup is near-identical to Colab but uses `/kaggle/working/` instead of Google Drive for persistence.

### Decision: DQN training extended to 3000 episodes on 20-node network
The larger network (20 nodes = 20 agents × 200 steps = 4000 agent decisions per episode) requires more training for epsilon to converge to 0.05. Epsilon decay set to 0.9990 (from 0.9985) to reach 0.05 after 3000 episodes. Training is slower than on 12 nodes due to CPU-bound simulation overhead — the T4 GPU and Kaggle CPUs provide minimal speedup as PyTorch forward passes are trivial compared to the Python simulation loop.

### Decision: Separate train/test for DQN
DQN trained for 3000 episodes, weights frozen, then evaluated for 500 episodes per run alongside heuristic conditions.

---

## Phase 4: RL Defender Design (Final)

### Algorithm: DQN (Deep Q-Network)

### Observation Space (Local Only)

| Feature | Description |
|---------|-------------|
| `detected_signal` | Probabilistic detection flag (0 or 1, p=0.09 if attacker present) |
| `suspicion_score` | Accumulated suspicion [0.0–1.0] |
| `inbox_alerts_norm` | Normalised count of inbox alerts [0.0–1.0] |
| `timestep_norm` | Current timestep normalised [0.0–1.0] |

### Action Space
| Action | ID | Description |
|--------|----|-------------|
| NOOP | 0 | Do nothing |
| SCAN | 1 | Scan own node |
| REIMAGE | 2 | Reimage own node |

### Reward Function (Final — after reward hacking fix)
| Event | Reward |
|-------|--------|
| Successful reimage (attacker removed) | +10 |
| Useful scan (attacker present) | +1 |
| NOOP on clean node | +0.5 |
| Unnecessary scan (node clean) | -0.1 |
| NOOP while attacker present | -2 |
| Unnecessary reimage (node clean) | -5 |

**Key change:** Unnecessary reimage penalty increased from -1 to **-5** to prevent reward hacking. NOOP with attacker increased from -1 to **-2**. Clean NOOP reduced from +1 to **+0.5**. These changes prevent the always-reimage degenerate policy discovered in earlier training runs.

### Neural Network Architecture
- Input: 4 features
- Hidden 1: 64 units, ReLU
- Hidden 2: 64 units, ReLU
- Output: 3 Q-values (one per action)
- Target network updated every 10 episodes

### Final Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 0.001 | Standard Adam LR |
| Gamma (discount) | 0.95 | Moderately future-oriented |
| Epsilon start | 1.0 | Full exploration initially |
| Epsilon min | 0.05 | 5% exploration at convergence |
| Epsilon decay | 0.9990 | Reaches ~0.05 after 3000 episodes |
| Batch size | 64 | Standard for replay buffer |
| Memory size | 10,000 | Sufficient for 200-step episodes |
| Target update | Every 10 episodes | Stable Q-value targets |
| Training episodes | 3000 | Full convergence on 20-node network |

---

## Bugs Found & Fixed

| Bug | Effect | Fix |
|-----|--------|-----|
| Integer node IDs | Defenders silently NOOPed every episode | Call `list(environment.network.nodes)` at runtime |
| NumPy 2.x incompatibility | `np.can_cast()` TypeError | Pin `numpy<2.0` |
| Zone.Identifier Windows artefacts | Module not found errors | Remove with `rm *.Identifier` |
| Epsilon decay too slow (0.995) | Epsilon stuck, agents never learned | Recalculate decay for target episode count |
| Nano tab/space mixing | TabError in edited scripts | Use Python `content.replace()` for all edits |
| Colab numpy binary incompatibility | `ValueError: numpy.dtype size changed` | Restart runtime after numpy install |
| `reset_episode()` wiping `_initialized` | Node agents destroyed each episode, epsilon stuck at 0.999 | Remove `_initialized = False` from `reset_episode()`; reset per-episode state only (suspicion, inbox) not the agents themselves |
| `time_to_first_compromise` always 0 | Start node always compromised at initialisation, threshold was `> 0` | Changed threshold to `> 1`; fallback changed from `steps` to `MAX_STEPS` |
| Reward hacking (always-reimage policy) | DQN converged to reimaging every node every step regardless of state; Q-value analysis confirmed REIMAGE dominated all states | Increased unnecessary reimage penalty from -1 to -5; increased NOOP-with-attacker penalty from -1 to -2; reduced clean NOOP reward from +1 to +0.5 |
| CyberBattleSim not found after Colab restart | Repo cloned into wrong subdirectory | Ensure install runs from correct path; verify with `pip install -e .` from repo root |

---

## Compute Environment

| Platform | Role | Notes |
|----------|------|-------|
| WSL2 Ubuntu 24.04 | Local development and file editing | Not used for training — CPU only |
| Google Colab (T4 GPU) | Initial DQN training attempts | Session timeouts caused repeated data loss; T4 GPU provides minimal speedup as bottleneck is CPU-bound Python simulation |
| Kaggle Notebooks | Final training and experiment runs | 30hr/week free compute, no idle timeout, sessions persist during active cell execution |

---

## Files Produced

```
/kaggle/working/CyberBattleSim/
├── run_experiment.py           — Main runner (6 conditions, 5 independent runs)
├── centralised_defender.py     — Centralised heuristic defender
├── decentralised_defender.py   — Decentralised multi-agent defender
├── dqn_defender.py             — DQN RL defender (final reward function)
├── train_dqn.py                — DQN training script (3000 episodes, 20 nodes)
├── analyse_results.py          — Statistical analysis + visualisations
├── CyberBattleSim_Colab.ipynb  — Google Colab notebook (legacy, superseded by Kaggle)
└── results/
    ├── run_1_results.csv  \
    ├── run_2_results.csv   |— Per-run raw data (500 eps × 6 conditions each)
    ├── run_3_results.csv   |
    ├── run_4_results.csv   |
    ├── run_5_results.csv  /
    ├── all_runs_results.csv    — Combined master dataset (run column added)
    ├── summary.csv             — Mean ± std per condition
    ├── dqn_agents.pkl          — Trained DQN weights (3000 episodes, 20 nodes)
    └── plots/
        ├── bar_reward.png
        ├── bar_compromised.png
        ├── box_reward.png
        └── timeseries_reward.png
```

---

## Research Questions — Current Status

| RQ | Question | Status |
|----|----------|--------|
| 1 | Can decentralised AI agents provide effective cyber defence? | **Answered — Yes, significantly (p≈0.000)** |
| 2 | Does cooperation between defenders improve detection and containment? | **Partially answered — final dataset pending** |
| 3 | How does decentralised compare with centralised defence? | **Answered — comparable, no significant difference** |
| 4 | Can reinforcement learning improve defensive decision policies? | **Answered in principle — final clean run pending** |

---

## Evaluation Metrics

| Metric | Description | Role |
|--------|-------------|------|
| `compromised_nodes` | Nodes with attacker present at episode end | Primary metric |
| `total_reward` | Cumulative attacker reward per episode | Secondary metric |
| `time_to_first_compromise` | Steps until attacker spreads beyond start node | Supplementary (now correctly recorded) |
| `steps` | Steps until termination | Supplementary |
| `terminated_early` | Whether attacker achieved goal before MAX_STEPS | Supplementary |

---

## Planned Next Steps

1. Complete DQN training (3000 episodes) on Kaggle — currently running
2. Back up trained model to `/kaggle/working/CyberBattleSim_Results/`
3. Run full 5 × 6-condition experiment on Kaggle
4. Run `analyse_results.py` on combined master dataset
5. Regenerate all plots to include DQN condition and all 5 runs
6. Update thesis Word document with final results and statistics
7. Write up methodology, results and discussion chapters
