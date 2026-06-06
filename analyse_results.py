"""
Results Analysis
=================
Reads all_runs_results.csv and produces:
  1. Mann-Whitney U tests (primary) + Cohen's d effect sizes
  2. Independent t-tests (supplementary)
  3. Bar charts — mean compromised nodes and mean reward
  4. Box plots — reward distribution per condition
  5. Time-series — rolling mean reward over episodes

Author: Louis Poole — BSc Computer Science & AI, Loughborough University
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

WORK_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(WORK_DIR, "results", "all_runs_results.csv")
OUTPUT_DIR   = os.path.join(WORK_DIR, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = defaultdict(lambda: {
    "total_reward":             [],
    "compromised_nodes":        [],
    "steps":                    [],
    "time_to_first_compromise": [],
})

with open(RESULTS_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cond = row["condition"]
        data[cond]["total_reward"].append(float(row["total_reward"]))
        data[cond]["compromised_nodes"].append(float(row["compromised_nodes"]))
        data[cond]["steps"].append(float(row["steps"]))
        data[cond]["time_to_first_compromise"].append(float(row["time_to_first_compromise"]))

LABELS = {
    "no_defender":               "No Defender",
    "centralised":               "Centralised",
    "decentralised_nocomms":     "Decent. (no alerts)",
    "decentralised_cooperative": "Decent. (cooperative)",
    "decentralised_competitive": "Decent. (competitive)",
    "rl_defender":               "RL Defender (DQN)",
}

COLOURS = {
    "no_defender":               "#e74c3c",
    "centralised":               "#3498db",
    "decentralised_nocomms":     "#2ecc71",
    "decentralised_cooperative": "#9b59b6",
    "decentralised_competitive": "#f39c12",
    "rl_defender":               "#1abc9c",
}

# Keep only conditions present in the data, in defined order
conditions = [c for c in LABELS if c in data]

log(f"Loaded data for {len(conditions)} conditions:")
for c in conditions:
    log(f"  {LABELS[c]}: {len(data[c]['total_reward'])} episodes")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    """Pooled Cohen's d effect size."""
    na, nb   = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2)
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return abs(np.mean(a) - np.mean(b)) / pooled_std


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ---------------------------------------------------------------------------
# 1. Statistical significance
# ---------------------------------------------------------------------------

log("\n" + "=" * 65)
log("STATISTICAL TESTS")
log("Primary  : Mann-Whitney U (non-parametric — all distributions non-normal)")
log("Secondary: Independent t-test")
log("Effect   : Cohen's d")
log("=" * 65)

baseline_reward      = data["no_defender"]["total_reward"]
baseline_compromised = data["no_defender"]["compromised_nodes"]

comparisons = [(c, "no_defender") for c in conditions if c != "no_defender"]
# Add cooperative vs competitive
if "decentralised_cooperative" in data and "decentralised_competitive" in data:
    comparisons.append(("decentralised_cooperative", "decentralised_competitive"))
# Add DQN vs centralised
if "rl_defender" in data and "centralised" in data:
    comparisons.append(("rl_defender", "centralised"))

for c_a, c_b in comparisons:
    label_a = LABELS.get(c_a, c_a)
    label_b = LABELS.get(c_b, c_b)
    log(f"\n{label_a} vs {label_b}:")

    for metric, label in [("compromised_nodes", "Compromised nodes"),
                           ("total_reward",      "Reward           ")]:
        a = data[c_a][metric]
        b = data[c_b][metric]

        # Mann-Whitney U (primary)
        u_stat, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        # t-test (supplementary)
        t_stat, p_t  = stats.ttest_ind(a, b)
        # Cohen's d
        d = cohens_d(a, b)

        log(f"  {label} | Mann-Whitney p={p_mw:.4f} {sig_stars(p_mw):<3} | "
            f"t-test p={p_t:.4f} {sig_stars(p_t):<3} | Cohen's d={d:.3f}")

log("\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

# Shapiro-Wilk normality check (on a sample to avoid size limitations)
log("\n" + "=" * 65)
log("NORMALITY CHECK — Shapiro-Wilk (sample n=200 per condition)")
log("=" * 65)
for c in conditions:
    sample = data[c]["total_reward"]
    if len(sample) > 200:
        sample = list(np.random.choice(sample, 200, replace=False))
    if len(sample) >= 3:
        stat, p = stats.shapiro(sample)
        log(f"  {LABELS[c]:<30} W={stat:.4f}  p={p:.4f}  "
            f"{'NORMAL' if p > 0.05 else 'NON-NORMAL'}")

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

labels  = [LABELS[c] for c in conditions]
colours = [COLOURS[c] for c in conditions]

mean_reward = [np.mean(data[c]["total_reward"])      for c in conditions]
std_reward  = [np.std(data[c]["total_reward"])       for c in conditions]
mean_comp   = [np.mean(data[c]["compromised_nodes"]) for c in conditions]
std_comp    = [np.std(data[c]["compromised_nodes"])  for c in conditions]
x = np.arange(len(conditions))

# ---------------------------------------------------------------------------
# 2. Bar chart — mean reward
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(x, mean_reward, yerr=std_reward, capsize=5,
              color=colours, edgecolor="black", linewidth=0.8, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
ax.set_ylabel("Mean Attacker Reward", fontsize=12)
ax.set_title("Mean Attacker Reward per Condition  (lower = better defence)\n"
             f"n = {len(data['no_defender']['total_reward'])} episodes per condition", fontsize=13)
ax.set_ylim(0, max(mean_reward) * 1.3)
for bar, mean in zip(bars, mean_reward):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{mean:.1f}", ha="center", va="bottom", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "bar_reward.png")
plt.savefig(out, dpi=150)
plt.close()
log(f"\nSaved: {out}")

# ---------------------------------------------------------------------------
# 3. Bar chart — mean compromised nodes
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(x, mean_comp, yerr=std_comp, capsize=5,
              color=colours, edgecolor="black", linewidth=0.8, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
ax.set_ylabel("Mean Compromised Nodes", fontsize=12)
ax.set_title("Mean Compromised Nodes per Condition  (lower = better defence)\n"
             f"n = {len(data['no_defender']['compromised_nodes'])} episodes per condition", fontsize=13)
ax.set_ylim(0, max(mean_comp) * 1.3)
for bar, mean in zip(bars, mean_comp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "bar_compromised.png")
plt.savefig(out, dpi=150)
plt.close()
log(f"Saved: {out}")

# ---------------------------------------------------------------------------
# 4. Box plots — reward distribution
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))
box_data = [data[c]["total_reward"] for c in conditions]
bp = ax.boxplot(box_data, patch_artist=True, notch=False)
for patch, colour in zip(bp["boxes"], colours):
    patch.set_facecolor(colour)
    patch.set_alpha(0.75)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
ax.set_ylabel("Attacker Reward", fontsize=12)
ax.set_title("Distribution of Attacker Reward per Condition", fontsize=13)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "box_reward.png")
plt.savefig(out, dpi=150)
plt.close()
log(f"Saved: {out}")

# ---------------------------------------------------------------------------
# 5. Time-series — rolling mean reward
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 6))
window = 20
for c in conditions:
    rewards = data[c]["total_reward"]
    rolling = [np.mean(rewards[max(0, i - window):i + 1]) for i in range(len(rewards))]
    ax.plot(rolling, label=LABELS[c], color=COLOURS[c], linewidth=2)
ax.set_xlabel("Episode (all runs combined)", fontsize=12)
ax.set_ylabel(f"Rolling Mean Reward (window={window})", fontsize=12)
ax.set_title("Attacker Reward Over Episodes — Rolling Mean", fontsize=13)
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "timeseries_reward.png")
plt.savefig(out, dpi=150)
plt.close()
log(f"Saved: {out}")

log(f"\nAll plots saved to {OUTPUT_DIR}")
