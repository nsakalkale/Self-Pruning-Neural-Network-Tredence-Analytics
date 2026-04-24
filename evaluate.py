"""
evaluate.py — Load results.json, produce all plots and print the report table.

Generates:
  • gate_distribution.png : histogram of final gate values per λ (the money plot)
  • training_curves.png : val acc + sparsity over training for each λ
  • tradeoff.png : pareto-style accuracy vs sparsity scatter
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


PALETTE = ["#4C72B0", "#DD8452", "#55A868"]   # muted blue / orange / green
RUNS_DIR = Path("./runs")


def load(path: Path = RUNS_DIR / "results.json"):
    if not path.exists():
        print(f"[!] {path} not found. Run train.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: gate value distributions
# ---------------------------------------------------------------------------

def plot_gate_distributions(results: list, out: Path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, res, color in zip(axes, results, PALETTE):
        gates = np.array(res["gate_values"])
        lam   = res["lambda"]
        spar  = res["final_sparsity"] * 100

        ax.hist(gates, bins=80, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_title(f"λ = {lam:.0e}\nsparsity = {spar:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate value (σ(score))", fontsize=9)
        ax.set_ylabel("Count" if ax == axes[0] else "", fontsize=9)
        ax.axvline(0.01, color="crimson", linestyle="--", linewidth=1.2, label="threshold=0.01")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Gate Value Distributions — Self-Pruning Network", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    from IPython.display import display
    display(fig)
    
    plt.close(fig)
    print(f"  ✓  {out}")


# ---------------------------------------------------------------------------
# Plot 2: training curves
# ---------------------------------------------------------------------------

def plot_training_curves(results: list, out: Path):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_acc  = fig.add_subplot(gs[0])
    ax_spar = fig.add_subplot(gs[1])

    for res, color in zip(results, PALETTE):
        lam     = res["lambda"]
        epochs  = [h["epoch"]   for h in res["history"]]
        val_acc = [h["val_acc"] * 100 for h in res["history"]]
        spars   = [h["sparsity"] * 100 for h in res["history"]]
        label   = f"λ={lam:.0e}"

        ax_acc.plot(epochs, val_acc, color=color, linewidth=2, label=label)
        ax_spar.plot(epochs, spars,  color=color, linewidth=2, label=label, linestyle="--")

    for ax, ylabel, title in [
        (ax_acc,  "Validation Accuracy (%)", "Accuracy over Training"),
        (ax_spar, "Sparsity (%)",            "Gate Sparsity over Training"),
    ]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Training Dynamics — Self-Pruning Network", fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight")

    from IPython.display import display
    display(fig)
    
    plt.close(fig)
    print(f"  ✓  {out}")


# ---------------------------------------------------------------------------
# Plot 3: accuracy–sparsity tradeoff
# ---------------------------------------------------------------------------

def plot_tradeoff(results: list, out: Path):
    fig, ax = plt.subplots(figsize=(6, 5))

    for res, color in zip(results, PALETTE):
        acc  = res["final_val_acc"] * 100
        spar = res["final_sparsity"] * 100
        lam  = res["lambda"]
        ax.scatter(spar, acc, s=120, color=color, zorder=5, label=f"λ={lam:.0e}")
        ax.annotate(f"λ={lam:.0e}\n({spar:.0f}%, {acc:.1f}%)",
                    xy=(spar, acc), xytext=(spar + 1, acc - 0.5),
                    fontsize=8, color=color)

    ax.set_xlabel("Sparsity (% dead weights)", fontsize=10)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax.set_title("Accuracy vs Sparsity Trade-off", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    
    from IPython.display import display
    display(fig)
    
    plt.close(fig)
    print(f"  ✓  {out}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_table(results: list):
    print(f"\n{'='*58}")
    print(f"  {'Lambda':<10} {'Val Acc':>10} {'Sparsity':>12} {'Dead / Total':>15}")
    print(f"  {'-'*54}")
    for r in results:
        dead  = r["dead_weights"]
        total = r["total_weights"]
        print(f"  {r['lambda']:<10.0e} {r['final_val_acc']*100:>9.2f}%"
              f" {r['final_sparsity']*100:>11.1f}%  {dead:>6}/{total}")
    print(f"{'='*58}\n")


def main():
    results = load()
    print_table(results)

    out_dir = RUNS_DIR
    plot_gate_distributions(results, out_dir / "gate_distribution.png")
    plot_training_curves(results,    out_dir / "training_curves.png")
    plot_tradeoff(results,           out_dir / "tradeoff.png")

    print("\nAll plots saved to ./runs/")


if __name__ == "__main__":
    main()