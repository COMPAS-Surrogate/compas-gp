"""Two actual COMPAS catalogue pilots, with the reference hidden until stopping."""
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from dataclasses import asdict
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('operational_runner',HERE/'run_study.py')
engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
import essential_runs as base
from posterior_convergence import event_target
from cosmic_integration.lnl_surrogate.stopping import symmetric_kl


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    out=HERE/'compas_v1';out.mkdir(exist_ok=False)
    engine.CONFIG.update(initial_sobol=128,boundary_points=98,batch=128,
                         candidate_points=4096,maximum_calls=2200)
    core=base.core
    loader=base.Experiment(out/'population','operational_stopping',False)
    source=base.HERE/'convergence_20260910/pilot'
    groups=[(100,'perfect'),(500,'uncertain')]
    files=[Path(__file__),HERE/'run_study.py',HERE/'PROTOCOL.md',
           engine.REPO/'src/cosmic_integration/lnl_surrogate/stopping.py']
    engine.write_json(out/'manifest.json',{'configuration':engine.CONFIG,'tolerances':asdict(engine.TOLERANCES),
        'groups':groups,'acquisition_seed':10117,'scope':'two preselected original catalogue pilots; not independent calibration',
        'posterior':'three shape coordinates plus exact conditional amplitude recovery in offline validation',
        'source_sha256':{str(p.relative_to(engine.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    for p in files:
        target=out/'source_snapshot'/p.relative_to(engine.REPO);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    outcomes=[]
    try:
        for n,mode in groups:
            folder=out/f'n{n}_{mode}';folder.mkdir()
            cat=np.load(source/f'catalogue_{n}_{mode}_0.npz')
            coeff,duration=cat['coeff'],float(cat['duration'])
            def evaluate(unit):
                phi=core.LOW[core.SHAPE]+unit*core.WIDTH[core.SHAPE]
                rates=np.array([loader.fast(p[0],p[1],.012,p[2]).ravel()[loader.active] for p in phi])
                return event_target(rates,coeff,duration)
            engine.write_json(out/'status.json',{'state':'running','current':folder.name,'phase':'blind_learning','completed':len(outcomes)})
            decision=engine.operational_run(engine.Evaluations(evaluate),10117,folder,dimensions=3)
            decision_bytes=(folder/'decision.json').read_bytes()
            # Only now load the independent cached reference maps and proposals.
            engine.write_json(out/'status.json',{'state':'running','current':folder.name,'phase':'post_decision_reference','completed':len(outcomes)})
            reference=[]
            for scramble in [0,1]:
                data=np.load(source/f'reference_n{n}_p16_{scramble}.npz')
                proposal=np.load(source/f'proposal_n{n}_p16_{scramble}.npz')
                phi,rates,logq=data['phi'],data['rates'],proposal['logq']
                exact=event_target(rates,coeff,duration)
                summary=loader.summarize(phi,rates,duration,len(coeff),core.TRUTH,exact-logq)
                unit=(phi-core.LOW[core.SHAPE])/core.WIDTH[core.SHAPE]
                reference.append((unit,phi,rates,logq,exact,summary))
            scores=[]
            for row in json.loads((folder/'records.json').read_text()):
                train=np.load(folder/f"checkpoint_{row['checkpoint']:03d}.npz")
                model,scaler=engine.fit(train['x'],train['lnl'])
                metrics=[];summaries=[]
                for unit,phi,rates,logq,exact,ds in reference:
                    pred=engine.predict(model,scaler,unit)
                    gs=loader.summarize(phi,rates,duration,len(coeff),core.TRUTH,pred-logq)
                    summaries.append(gs)
                    metrics.append({**core.compare(ds,gs),**{k:v for k,v in core.posterior_metrics(exact-logq,pred-logq,phi).items() if k.startswith('kl_')}})
                numeric=base.precision(reference[0][-1],reference[1][-1])[0] and base.precision(*summaries)[0]
                scores.append({'checkpoint':row['checkpoint'],'calls':row['calls'],
                    'reference_kl':[max(m['kl_direct_to_gp'],m['kl_gp_to_direct']) for m in metrics],
                    'accuracy_pass':base.accuracy(metrics),'numerical_pass':numeric,'metrics':metrics})
            score={'decision_sha256':hashlib.sha256(decision_bytes).hexdigest(),'scores':scores,
                   'premature_stop':decision['state']=='operational_stop' and not scores[-1]['accuracy_pass'],
                   'reference_source':str(source),'final_numerical_pass':scores[-1]['numerical_pass']}
            assert decision_bytes==(folder/'decision.json').read_bytes()
            engine.write_json(folder/'reference_score.json',score)
            outcomes.append({'family':folder.name,'seed':10117,**decision,'premature_stop':score['premature_stop'],
                             'reference_numerical_pass':score['final_numerical_pass'],
                             'final_reference_kl':max(scores[-1]['reference_kl'])})
            engine.write_json(out/'outcomes.json',outcomes)
        engine.write_json(out/'status.json',{'state':'complete','completed':len(outcomes)})
    except Exception as exc:
        engine.write_json(out/'status.json',{'state':'failed','completed':len(outcomes),'error':repr(exc)})
        raise


if __name__=='__main__':main()
