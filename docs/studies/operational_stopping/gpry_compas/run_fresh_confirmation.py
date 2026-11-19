"""Frozen fresh-catalogue confirmation, followed by independent endpoint scoring."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import time
import numpy as np
import compare_random as train
import run_stable_scheduled_audit as audit
import score_scheduled_audit as validation
from cosmic_integration.lnl_surrogate.scheduled_audit import AuditPolicy

HERE = Path(__file__).resolve().parent
BASE = train.base

def groups() -> list[tuple]:
    """Fresh case identifiers alter catalogue, design and audit RNGs together."""
    return [(t,n,m,c,s) for t in range(3) for n in (100,1000)
            for c in (100,101) for m in ('perfect','uncertain') for s in ('bo','random')]

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); out=args.outdir.resolve()
    out.mkdir(parents=True,exist_ok=args.resume)
    sources=[Path(__file__),HERE/'CONFIRMATION_PROTOCOL.md',HERE/'compare_random.py',
             HERE/'run_experiment.py',HERE/'run_stable_scheduled_audit.py',
             HERE/'score_scheduled_audit.py',HERE/'score_comparison.py',HERE/'check_precision.py',
             *sorted((BASE.REPO/'src/cosmic_integration').rglob('*.py'))]
    hashes={str(p.relative_to(BASE.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    audit.POLICY=AuditPolicy(sizes=(96,))
    manifest=dict(source_sha256=hashes,policy=asdict(audit.POLICY),groups=groups(),maximum=658,
                  scope='Fresh fixed-injection confirmation; paired measurement views, not SBC')
    manifest=json.loads(json.dumps(manifest))
    if args.resume:
        assert json.loads((out/'manifest.json').read_text())==manifest
    else:
        BASE.write_json(out/'manifest.json',manifest)
        for p in sources:
            q=out/'source_snapshot'/p.relative_to(BASE.REPO);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    tr=out/'training'; ar=out/'audits'; tr.mkdir(exist_ok=True);ar.mkdir(exist_ok=True)
    loader=None; decisions={}; trained={}
    for ti,n,mode,case,strategy in groups():
        name=f't{ti}_n{n}_{mode}_c{case}_{strategy}'
        path=tr/name/'decision.json'
        if path.exists(): d=json.loads(path.read_text())
        else:
            if (tr/name).exists():
                # Preserve an interrupted trajectory; restart it deterministically.
                (tr/name).rename(tr/f'{name}_interrupted_{time.time_ns()}')
            if loader is None:loader=BASE.data_source.Experiment(out/f'population_{time.time_ns()}','fresh_confirmation_no_reference',False)
            d=train.run_case(tr,loader,ti,n,mode,case,strategy,658)
        trained[name]=d
        path=ar/f'{name}.json'
        if path.exists(): a=json.loads(path.read_text())
        else:
            if loader is None:loader=BASE.data_source.Experiment(out/f'population_{time.time_ns()}','fresh_confirmation_no_reference',False)
            a=audit.run_case(tr,ar,loader,name,d)
            a['cohort']='fresh_confirmation';BASE.write_json(path,a)
        decisions[name]=a
        BASE.write_json(out/'decisions.json',decisions)
        BASE.write_json(out/'status.json',dict(state='running',phase='training_and_blind_audit',completed=len(decisions),expected=48,last_case=name))
    frozen=(out/'decisions.json').read_bytes(); sha=hashlib.sha256(frozen).hexdigest()
    BASE.write_json(out/'status.json',dict(state='decisions_frozen',completed=48,decision_sha256=sha))
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
