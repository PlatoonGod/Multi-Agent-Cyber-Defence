# An Internet of AI for Cyber Security: A Decentralised Multi-Agent Defence Simulation

**Author:** Louis Poole — BSc Computer Science \& AI, Loughborough University, 2025–2026
**Supervisor:** Dr Andrea Soltoggio

Source code and data accompanying the thesis of the same title.

## Environment

* Python 3.12
* Tested on Kaggle Notebooks (CPU runtime) and WSL2 Ubuntu 24.04

## Installation

CyberBattleSim is not on PyPI and must be installed from source:

```bash
git clone https://github.com/microsoft/CyberBattleSim.git
cd CyberBattleSim
pip install -e .
cd ..
pip install -r requirements.txt
```

**Important:** NumPy must be pinned below 2.0 (see thesis §3.7, Challenge 2).

## Reproducing the results

1. **Train the DQN** (3,000 episodes; \~8 hours on Kaggle CPU, checkpointed every 500):

```bash
   python src/train\\\_dqn.py
```

Produces `results/dqn\\\_agents.pkl`. (A pre-trained copy is included — skip
this step if you want to go straight to evaluation.)

2. **Run all 6 conditions × 5 independent runs**:

```bash
   python src/run\\\_experiment.py
```

Produces `results/run\\\_{1..5}\\\_results.csv` and `results/all\\\_runs\\\_results.csv`.

3. **Analyse and plot**:

```bash
   python src/analyse\\\_results.py
```

Prints Mann-Whitney U tests, Cohen's d, and Shapiro-Wilk normality checks
(reproduces Tables 8 and 9 in the thesis). Writes plots to
`results/plots/` (Figures 6, 7, 10, 11 in the thesis).

## File layout

See Appendix A of the thesis for per-file descriptions.

