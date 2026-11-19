"""Summarize a retrieved campaign snapshot and generate its preliminary TeX table."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    rows=[]
    for path in sorted((root/'retrieved/runs').glob('*/score.json')):
        record=json.loads(path.read_text());job=record['job'];last=record['attempts'][-1]
        done=json.loads((path.parent/'complete.json').read_text())
        checkpoint=path.parent/f"checkpoint_{done['labels']}.npz"
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest!=done['checkpoint_sha256']:raise ValueError(f'Checkpoint mismatch: {checkpoint}')
        assert record['labels']==done['labels'] and record['outcome']==last['outcome']
        if record['outcome']=='accurate':
            assert all(s['accuracy_pass'] for s in last['scores'])
            assert last['kl_precision'] and last['direct_precision']['passed'] and last['gp_precision']['passed']
        rows.append(dict(name=job['name'],catalogue=job['catalogue'],strategy=job['strategy'],
            group=job['group'],labels=record['labels'],outcome=record['outcome'],
            maximum_directed_kl=max(max(s['directed_kl']) for s in last['scores']),
            score_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=digest))
    summary=dict(retrieved_scored=len(rows),total_planned=632,
        outcomes=dict(Counter(r['outcome'] for r in rows)),rows=rows,
        scope='Incomplete scheduler-ordered snapshot; no representative success-rate or evaluation-savings inference.')
    (root/'results_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=['# Preliminary four-input paper results','',
          f"Retrieved and verified {len(rows)} scored fits out of 632 planned.",
          'These are final-budget comparisons, not estimates of evaluations required.','',
          '| Case | Outcome | Maximum directed KL |','|---|---|---:|']
    text += [f"| {r['name']} | {r['outcome']} | {r['maximum_directed_kl']:.6g} |" for r in rows]
    text += ['',summary['scope']]
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--tex',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.root)
    by_name={r['name']:r for r in result['rows']}
    table=[]
    for n in (10,50):
        for mode in ('perfect','uncertain'):
            cells=[]
            for strategy in ('bo','random'):
                name=f'count_n{n}_{mode}_c200_s17_{strategy}'
                row=by_name.get(name)
                cells.append('---' if row is None else f"{row['maximum_directed_kl']:.5f}")
            table.append(f"{n} & {mode.capitalize()} & {' & '.join(cells)} \\\\")
    args.tex.write_text('\n'.join([r'\begin{tabular}{rlrr}', r'\hline',
        r'Expected count & Measurements & BO & Random \\', r'\hline',
        *table, r'\hline', r'\end{tabular}'])+'\n')


if __name__=='__main__':main()
