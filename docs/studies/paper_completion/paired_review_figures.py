"""Publication formatting for the generated paired-data illustration.

Keep this rendering pass separate from the running campaign's frozen sources.
No catalogue, likelihood, or acquired point is changed.
"""
from pathlib import Path
import sys
import matplotlib.ticker as ticker

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'event_uncertainty'))
import paired_diagnostics as diagnostics


def main() -> None:
    original_save = diagnostics.save
    def publication_save(fig, path: Path) -> None:
        for ax in fig.axes:
            ax.set_xticks([5, 10, 20, 40], ['5', '10', '20', '40'])
            ax.set_yticks([.05, .1, .2, .4], ['0.05', '0.1', '0.2', '0.4'])
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        original_save(fig, path)
    diagnostics.save = publication_save
    diagnostics.plot_data(HERE.parent / 'event_uncertainty/paired_20260912/pilot',
                          REPO / 'overleaf/figures/bo_study')


if __name__ == '__main__':
    main()
