"""Plot fixed-population agreement and the exploratory noisy-mock pilot."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

def main():
    data = np.load(HERE/'reduced_seed_results_512M_anchored/seed_17/validation.npz')
    cases = json.loads((HERE/'reduced_calibration_results/cases.json').read_text())
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    labels = [r'$\alpha$', r'$\sigma$', r'$a_{\mathrm{SFR}}$', r'$d_{\mathrm{SFR}}$']
    for i, ax in enumerate(axes[0]):
        bins = np.linspace(data['draws'][:, i].min(), data['draws'][:, i].max(), 35)
        for key, color, style, label in [('direct_weights', '#222222', '-', 'Direct likelihood'), ('gp_weights', '#326bb0', '--', 'Repaired reduced GP')]:
            ax.hist(data['draws'][:, i], bins=bins, weights=data[key], density=True,
                    histtype='step', color=color, ls=style, lw=2, label=label)
        ax.set_xlabel(labels[i])
    axes[0, 0].set_ylabel('Posterior density')
    axes[0, 1].legend(frameon=False, fontsize=9)
    for i, ax in enumerate(axes[1]):
        ax.scatter([c['events'] for c in cases], [c['width68_ratio'][i] for c in cases], color='#326bb0', s=28)
        ax.axhline(1, color='#222222', lw=1)
        ax.set_ylim(.975, 1.025)
        ax.set_xlabel('Events in noisy mock')
        ax.set_title(labels[i])
    axes[1, 0].set_ylabel('GP / direct 68% interval width')
    fig.suptitle('Repaired GP reproduces the direct constraints\nTop: 81 events, fixed 512M population • Bottom: 20 mocks, disjoint 256M halves', fontsize=15)
    fig.savefig(HERE/'reduced_calibration_results/validation_overview.png', dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    main()
