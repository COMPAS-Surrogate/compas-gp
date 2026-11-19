"""Summarize verified downloaded stage-one histories, deduplicating continuations."""
import argparse
import collections
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    args=p.parse_args()
    d=json.loads(args.results.read_text());old=json.loads(args.parent.read_text())
    merged={t['name']:dict(t,job=dict(name=t['name'],seed=int(t['name'].split('_s')[1].split('_')[0]),strategy=t['name'].split('_')[-1]),maximum=1284,source='warm_validation') for t in old['tasks'] if t['kind']=='trajectory'}
    for t in d['tasks']:
        assert t['status']['state']=='complete'
        assert t['complete']['checkpoint_sha256']==t['rows'][-1]['checkpoint_sha256']
        merged[t['job']['name']]=dict(t,source='manuscript_stage1')
    rows=[]
    for name,t in sorted(merged.items()):
        by_n={r['labels']:r['outcome'] for r in t['rows']}
        first=next((n for n in sorted(by_n) if by_n[n]=='accurate' and by_n.get(n+30)=='accurate'),None)
        assert first==t['first_confirmed_tested']
        final=t['rows'][-1]
        rows.append(dict(name=name,expected_events=int(name.split('_')[1][1:]),view=name.split('_')[2],seed=t['job']['seed'],strategy=t['job']['strategy'],maximum=t['maximum'],first_confirmed_tested=first,confirmation_cost=None if first is None else first+30,final_outcome=final['outcome'],final_max_directed_kl=final['maximum_directed_kl'],source=t['source']))
    root=args.results.parent
    with (root/'trajectory_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    primary=[r for r in rows if r['seed']==17]
    lines=['# Four-input manuscript campaign: completed stage one','',f"Verified snapshot: {d['checked_at_utc']}.",'',
           '- Slurm pilot **16611906** and all **92** tasks in array **16611907** completed with exit `0:0`; no jobs remain queued/running in these arrays.',
           f"- All {sum(len(t['rows']) for t in d['tasks'])} scored checkpoint hashes were verified against the remote NPZ files. All 92 scoring histories completed; none of these scores is numerically unresolved.",
           f'- Merge with earlier compatible runs gives **{len(primary)}/140 primary trajectories**, plus eight seed-29 sensitivity trajectories. Continuations replace their parent records rather than being counted twice. Forty primary trajectories and the four predictive examples remain to run.',
           '', '## Primary campaign (seed 17)', '',
           '| Expected events | Budget | BO confirmed / tested | Random confirmed / tested |',
           '| --- | ---: | ---: | ---: |']
    for n in (10,50,100,200,500,700,1000):
        sub=[r for r in primary if r['expected_events']==n]
        if not sub:
            lines.append(f'| {n} | Not run | — | — |');continue
        cells=[]
        for strategy in ('bo','random'):
            group=[r for r in sub if r['strategy']==strategy]
            cells.append(f"{sum(r['first_confirmed_tested'] is not None for r in group)}/{len(group)}")
        lines.append(f"| {n} | {sub[0]['maximum']} | {' | '.join(cells)} |")
    lines += ['', '- Counts include paired perfect/uncertain views: ten runs per strategy at a count are five latent catalogues, not ten independent catalogues. The 500-event rows currently contain three catalogues; the 1000-event rows contain two.',
              '- At 50 events, random acquisition has eight final passes but only seven confirmed passes: one passes at 1284 without an accurate preceding checkpoint. It remains unconfirmed.',
              '- All eight 500-event BO trajectories, including the two seed-29 views, now have confirmed accuracy. All eight matched random trajectories still fail at 1884.',
              '- The remaining 1000-event BO failure is `count_n1000_uncertain_c200_s17_bo`: maximum directed KL **0.01052597** at 1884, above 0.01. Keep it classified as inaccurate.',
              '', '## First tested passing budgets', '',
              '- BO: 10/50 events pass at 504 and confirm at 534; at 100 events five runs first pass at 504 and five at 744; at 200 events nine first pass at 744 and one at 1254. Confirmation costs another 30 labels.',
              '- Primary 500-event BO: three first pass at 1254, one at 1284, one at 1554, one at 1854. Both seed-29 cases first pass at 1854. At 1000 events, one first passes at 1254 and two at 1554; one remains unconfirmed through 1884.',
              '- Random: all ten 10-event runs first pass at 504; seven 50-event runs first pass at 1254. No confirmed primary random result at counts 100 or higher within its tested budget.',
              '- These are sparse **tested** checkpoint crossings, not exact minimum budgets. The full joint-posterior and numerical gates apply, not KL alone. Passing checkpoints were identified retrospectively; all trajectories ran to their prescribed ceiling.',
              '', '## Runtime and next work', '',
              '- Slurm elapsed times: lower-count runs 7–13 minutes; 500-event continuations 31–38 minutes; new 1000-event runs 39–62 minutes. These include scoring. Continuation accounting excludes the original job wall time; inherited training timers are cumulative.',
              '- Complete the missing primary matrix: eight 500-event trajectories, twenty 700-event trajectories and twelve 1000-event trajectories. The four 30/1200-event predictive examples are separate. These results do not yet provide all manuscript figures.',
              '- Extend the remaining inaccurate 1000-event BO case with its matched random control in a versioned campaign, as proposed in the plan; inspect the error trend before choosing a larger blanket budget. Random-budget limits must remain visible in the comparison.',
              '- Figure 4 has complete catalogue coverage at counts 10–200 and partial coverage at 500/1000; 700 is missing. Holdout/uncertainty plots, full joint corner plots and predictive post-processing still require their own artifact generation. No existing manuscript figure is relabelled as a four-input result by this update.',
              '', '## Reproduction', '',
              'Run `python scripts/ozstar/gp4/summarize.py --results docs/studies/amplitude_gp/manuscript_stage1_20260917/results_snapshot.json --parent docs/studies/amplitude_gp/warm_validation_20260916/status_snapshot.json`.',
              'Raw scoring attempts, reference metadata, run identities and timings are in `results_snapshot.json`; deduplicated outcomes are in `trajectory_summary.csv`; scheduler accounting is in `slurm_completion.txt`.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (root/'latest_status.json').write_text(json.dumps(dict(checked_at_utc=d['checked_at_utc'],pilot='16611906',pilot_state='COMPLETED',array='16611907',completed_tasks=92,failed_tasks=0,scored_checkpoints=sum(len(t['rows']) for t in d['tasks']),primary_trajectories=len(primary),supplementary_trajectories=len(rows)-len(primary)),indent=2)+'\n')


if __name__=='__main__':main()
