"""Independent scoring of already frozen fresh confirmation decisions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import compare_random as train
import score_scheduled_audit as validation
BASE=train.base
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args();out=args.input.resolve();tr=out/'training';loader=None
    frozen=(out/'decisions.json').read_bytes();sha=hashlib.sha256(frozen).hexdigest()
    status=json.loads((out/'status.json').read_text());assert status['decision_sha256']==sha
    decisions=json.loads(frozen);assert len(decisions)==48 and all(not d['reference_accessed'] for d in decisions.values())
    BASE.write_json(out/'scoring_manifest.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),decision_sha256=sha,repair='Use existing score_checkpoint API; no decision or tolerance change'))
    # References first accessed here, after every operational decision is frozen.
    scored=[]; outcomes=[]; cache={}; direct_cache={}; refhash={}; offline=out/'offline';offline.mkdir(exist_ok=True)
    for name,d in decisions.items():
        ti,n=d['truth_index'],d['expected_events']; key=(ti,n)
        cat=dict(np.load(tr/name/'catalogue.npz'));truth=train.TRUTHS[ti]
        catkey=(ti,n,d['mode'],d['case'])
        if catkey not in direct_cache:
            if key not in cache:cache[key]=validation.references(HERE/'extension_v2',ti,n,refhash)
            refs=cache[key]; checked=validation.score.direct_checks(refs,cat,truth)
            if not checked[-1]['passed']:
                if loader is None:loader=BASE.data_source.Experiment(out/f'population_{time.time_ns()}','confirmation_reference_refinement',False)
                fisher=BASE.core.fisher_matrix(lambda t:loader.fast(*t),truth,BASE.core.WIDTH,float(cat['duration']))
                for power in (17,18):
                    refs=[]
                    for scramble in (0,1):
                        seed=202609211+100*ti+n+10*d['case']+scramble
                        phi,logq=BASE.data_source.reference_design(truth,fisher,seed,power)
                        f=offline/f'reference_t{ti}_n{n}_c{d["case"]}_p{power}_{scramble}.npz'
                        if f.exists(): rates=np.load(f)['rates']
                        else:
                            rates=np.array([loader.fast(p[0],p[1],.012,p[2]).ravel()[loader.active] for p in phi]);np.savez(f,phi=phi,rates=rates,logq=logq)
                        refs.append((phi,rates,logq))
                    checked=validation.score.direct_checks(refs,cat,truth)
                    if checked[-1]['passed']:break
            direct_cache[catkey]=(refs,checked)
        refs,checked=direct_cache[catkey]; selected=d['rules']['fixed96']; bycount={}
        for count in sorted({658}|({selected['training']} if selected else set())):
            f=offline/f'{name}_{count}.json'
            if f.exists(): row=json.loads(f.read_text())
            else:
                meta={k:d[k] for k in ('name','truth_index','expected_events','mode','case','strategy')};meta['points']=count
                row=validation.score_checkpoint(tr/name/f'checkpoint_{count}.npz',refs,cat,truth,checked,meta);BASE.write_json(f,row)
            scored.append(row);bycount[count]=row
        row=bycount[selected['training'] if selected else 658]
        numerical=row['reference_numerical'] and row['gp_numerical'] and row['four_numerical']
        outcome='no_stop' if selected is None else 'unresolved' if not numerical else 'accurate' if row['accuracy_pass'] else 'premature'
        outcomes.append(dict(name=name,strategy=d['strategy'],expected_events=n,mode=d['mode'],truth_index=ti,case=d['case'],selected=selected,
            outcome=outcome,maximum_directed_kl=row['maximum_directed_kl'],numerical=bool(numerical),final_accuracy=bycount[658]['accuracy_pass'],
            total_cost=selected['total'] if selected else 658+d['audit_labels']['fixed96']))
        BASE.write_json(offline/'records.json',scored);BASE.write_json(offline/'outcomes.json',outcomes)
        BASE.write_json(out/'status.json',dict(state='running',phase='independent_scoring',scored=len(outcomes),expected=48,decision_sha256=sha))
    assert (out/'decisions.json').read_bytes()==frozen
    BASE.write_json(offline/'reference_hashes.json',refhash)
    BASE.write_json(out/'status.json',dict(state='complete',completed=48,scored=48,decision_sha256=sha))
if __name__=='__main__':main()
