"""Static P-P and posterior predictive plots for the frozen response experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {'direct': '#262626', 'gp': '#346bb3'}
LABELS = [r'$\alpha$', r'$\sigma$', r'$a_{\rm SFR}$', r'$d_{\rm SFR}$']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='uncertain_100_results')
    args = parser.parse_args()
    root = HERE/args.outdir
    rows = [r for r in json.loads((root/'cases.json').read_text()) if r['mode'] == 'uncertain']
    valid = [r for r in rows if r['status'] == 'evaluated']
    n = len(valid)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), constrained_layout=True)
    x = np.linspace(0, 1, 401)
    # DKW gives a simultaneous 95% envelope over CDF thresholds for each panel;
    # this is not a simultaneous four-parameter certification.
    epsilon = np.sqrt(np.log(2/.05)/(2*n))
    for j, ax in enumerate(axes):
        ax.fill_between(x, np.maximum(0, x-epsilon), np.minimum(1, x+epsilon), color='.9', label='95% per-panel DKW band')
        ax.plot(x, x, color='.55', lw=1)
        for name, style in [('direct', '-'), ('gp', '--')]:
            ranks = np.sort([r['summaries'][0][name]['ranks'][j] for r in valid])
            ax.step(np.r_[0, ranks, 1], np.r_[0, np.arange(1, n+1)/n, 1], where='post',
                    color=COLORS[name], ls=style, label={'direct': 'Direct', 'gp': 'GP'}[name])
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Posterior CDF at injected value', title=LABELS[j])
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Fraction of catalogue ranks below threshold')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Posterior ranks with synthetic measurement uncertainty\n{n}/{len(rows)} evaluated catalogues • matched population • 482 labels • categorical response', fontsize=13)
    for ext in ['png', 'pdf']:
        fig.savefig(root/f'pp_plot.{ext}', dpi=180)
    plt.close(fig)

    first = valid[:3]
    fig, axes = plt.subplots(len(first), 3, figsize=(11, 3*len(first)+.5), squeeze=False, constrained_layout=True)
    labels = ['Replicated event count', 'Replicated mean mass-bin index', 'Replicated mean redshift-bin index']
    for row_index, record in enumerate(first):
        case = record['case']
        path = root/f'uncertain_{case:03d}'/'predictive.json'
        if not path.exists():
            raise ValueError('First three retained cases do not have predictive diagnostics; do not silently substitute cases')
        ppc = json.loads(path.read_text())
        for j, ax in enumerate(axes[row_index]):
            samples = {name: np.array(data['replicate_statistics'])[:, j] for name, data in ppc.items()}
            pooled = np.concatenate(list(samples.values()))
            bins = (np.arange(np.floor(pooled.min())-.5, np.ceil(pooled.max())+1.5)
                    if j == 0 else np.linspace(pooled.min()-.01, pooled.max()+.01, 25))
            for name, style in [('direct', '-'), ('gp', '--')]:
                ax.hist(samples[name], bins=bins, density=True, histtype='step', color=COLORS[name],
                        ls=style, lw=1.5, label={'direct': 'Direct', 'gp': 'GP'}[name])
            ax.axvline(ppc['direct']['observed_statistics'][j], color='#ba712c', ls=':', lw=1.5, label='Observed')
            ax.set_xlabel(labels[j])
            ax.spines[['top', 'right']].set_visible(False)
        axes[row_index, 0].set_ylabel(f'Case {case}: density\nN = {record["events"]}')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Posterior predictive replicated catalogues\nFirst three cases • 1,000 replicates per method • measurement response and Poisson variation included', fontsize=13)
    for ext in ['png', 'pdf']:
        fig.savefig(root/f'posterior_predictive.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(first), 2, figsize=(11, 3*len(first)+.5), squeeze=False, constrained_layout=True)
    for row_index, record in enumerate(first):
        ppc = json.loads((root/f'uncertain_{record["case"]:03d}'/'predictive.json').read_text())
        for ax, coordinate in zip(axes[row_index], ['mass', 'z']):
            for name, style in [('direct', '-'), ('gp', '--')]:
                q = np.array(ppc[name][f'{coordinate}_count_quantiles'])
                x = np.arange(q.shape[1])
                ax.fill_between(x, q[0], q[2], color=COLORS[name], alpha=.12, step='mid')
                ax.step(x, q[1], where='mid', color=COLORS[name], ls=style,
                        label={'direct': 'Direct median / 90% band', 'gp': 'GP median / 90% band'}[name])
            observed = ppc['direct'][f'observed_{coordinate}_counts']
            ax.plot(x, observed, '.', color='#ba712c', ms=4, label='Observed')
            ax.set_xlabel('Measured mass-bin index' if coordinate == 'mass' else 'Measured redshift-bin index')
            ax.set_ylim(bottom=0)
            ax.spines[['top', 'right']].set_visible(False)
        axes[row_index, 0].set_ylabel(f'Case {record["case"]}: event count')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Posterior predictive marginal count distributions\nFirst three cases • pointwise 90% replicated-count bands • categorical measurement coordinates', fontsize=13)
    for ext in ['png', 'pdf']:
        fig.savefig(root/f'predictive_marginals.{ext}', dpi=180)
    plt.close(fig)

    # Fixed first catalogue: show where acquisition spent the 256 extra labels.
    training = np.load(root/'uncertain_000'/'training.npz')
    points = training['x']
    first_case = next(r for r in rows if r['case'] == 0)
    truth = np.array(first_case['truth'])[[0, 1, 3]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    shape_labels = [LABELS[0], LABELS[1], LABELS[3]]
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(points[:226, i], points[:226, j], s=13, facecolors='none',
                   edgecolors='.6', linewidths=.6, label='226 initial / boundary points')
        ax.scatter(points[226:, i], points[226:, j], s=9, color=COLORS['gp'], alpha=.6,
                   label='256 acquired points')
        ax.plot(truth[i], truth[j], 'x', color='#ba712c', ms=9, mew=2, label='Injected parameters')
        ax.set(xlabel=shape_labels[i], ylabel=shape_labels[j])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, loc='best')
    fig.suptitle(f'Parameter locations selected for likelihood evaluations\nCase 0 • N = {first_case["events"]} • two 128-point likelihood-variance batches with 10% uniform exploration', fontsize=13)
    for ext in ['png', 'pdf']:
        fig.savefig(root/f'acquired_parameters.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
