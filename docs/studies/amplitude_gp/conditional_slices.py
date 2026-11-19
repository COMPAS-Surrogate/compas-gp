"""Test amplitude slices of trained four-input GPs against the exact Gamma law.

Other coordinates are fixed at the catalogue injection for this diagnostic only.
No injection coordinate is supplied to acquisition or used to narrow a prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import small_batch_followup as study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    root = args.input.resolve()
    amplitude = np.linspace(.005, .015, 32769)
    output = []
    for job in study.jobs():
        folder = root/job['name']
        if not (folder/'complete.json').exists():
            continue
        cat = dict(np.load(study.BASELINE/f'catalogue_n{job["expected"]}_{job["mode"]}.npz'))
        # The frozen catalogue exposure gives E[N]=expected at this injection.
        # Linearity therefore fixes the conditional Gamma rate independently
        # of the GP or the event measurement view.
        beta = job['expected']/float(cat['truth'][2])
        exact = len(cat['coeff'])*np.log(amplitude)-beta*amplitude
        points = np.tile(cat['truth'], (len(amplitude), 1))
        points[:, 2] = amplitude
        for labels in (414, 744):
            state, _, _ = study.load_state(folder/f'checkpoint_{labels}.npz')
            predicted = study.pilot.base.predict(state, (points-study.pilot.LOW)/study.pilot.WIDTH)
            result = study.pilot.score(amplitude[:, None], exact, predicted, np.zeros(len(amplitude)))
            row = dict(**job, labels=labels, observed=len(cat['coeff']), **result)
            output.append(row)
            np.savez_compressed(folder/f'conditional_slice_{labels}.npz', amplitude=amplitude,
                                exact=exact, gp=predicted)
    study.pilot.base.write_json(root/'conditional_slices.json', output)
    print(json.dumps(dict(completed_trajectories=len(output)//2,
        accurate_by_budget={str(n): sum(r['accuracy_pass'] for r in output if r['labels'] == n)
                            for n in (414, 744)}), indent=2))


if __name__ == '__main__':
    main()
