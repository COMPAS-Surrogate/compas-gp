"""Export held-out diagnostics, without presenting reweighting as a reference."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out = Path(__file__).parent / "results_seed17"
    result = json.loads((out / "summary.json").read_text())
    with np.load(out / "heldout.npz") as f:
        n = int(f["posterior_count"])
        theta, direct, gp = f["theta"][:n], f["direct_lnl"][:n], f["gp_lnl"][:n]
    finite = np.isfinite(direct) & np.isfinite(gp)
    offset = np.mean(gp[finite]-direct[finite])
    residual = gp[finite]-direct[finite]-offset
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    axes[0].scatter(direct[finite], gp[finite]-offset, s=24, color="#2563a6")
    limits = [min(direct[finite].min(), (gp[finite]-offset).min()),
              max(direct[finite].max(), (gp[finite]-offset).max())]
    axes[0].plot(limits, limits, "--", color="#414141", lw=1)
    axes[0].set(xlabel="Direct log likelihood", ylabel="GP log likelihood (offset aligned)",
                title="Held-out GP posterior locations")
    axes[1].scatter(theta[finite, 0], residual, s=24, color="#2563a6")
    axes[1].axhspan(-.1, .1, color="#dce8f3")
    axes[1].axhline(0, color="#414141", ls="--", lw=1)
    axes[1].set(xlabel="alpha", ylabel="Centred GP minus direct lnL", title="Likelihood-shape errors")
    with np.load(out / "training.npz") as f:
        raw = f["raw_lnl"]
        ref = float(f["initial_reference"])
    axes[2].plot(np.arange(len(raw))+1, np.maximum.accumulate(raw), color="#2563a6")
    axes[2].axhline(ref, color="#cb741a", ls="--", label="Initial transform ceiling")
    axes[2].set(xlabel="Training evaluation", ylabel="Best direct log likelihood",
                title="Peak discovery versus transform ceiling")
    axes[2].legend(fontsize=9)
    fig.suptitle("Joint 32M GP pilot: 81 events, fixed coarse grid, seed 17\n"
                 "Held-out rejection check; no direct-reference posterior computed", fontsize=12)
    fig.savefig(out / "heldout_diagnostics.png", dpi=170)
    plt.close(fig)
    if (out / "local_refinement/summary.json").exists():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        for ax, folder, label in zip(axes, [out, out / "local_refinement"],
                                     ["256 training points", "356 training points (local refinement)"]):
            with np.load(folder / "heldout.npz") as f:
                error = f["gp_lnl"][:70]-f["direct_lnl"][:70]
                error -= error.mean()
                alpha = f["theta"][:70, 0]
            ax.axhspan(-.1, .1, color="#dce8f3", label="Target: +/-0.1 lnL")
            ax.scatter(alpha, error, s=24, color="#2563a6")
            ax.axhline(0, color="#414141", ls="--", lw=1)
            ax.set(xlabel="alpha", ylabel="Centred GP minus direct lnL", ylim=(-11, 11),
                   title=f"{label}\nRMS = {np.sqrt(np.mean(error**2)):.2f} lnL")
        axes[0].legend(fontsize=9)
        fig.suptitle("Joint GP pilot: fresh posterior-location checks\n"
                     "Separate 70-point validation sets; same 32M population and 81-event catalogue", fontsize=12)
        fig.savefig(out / "refinement_comparison.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
