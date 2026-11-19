"""Replay saved audit outcomes and verify pairing, costs and checkpoint identity."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'operational_stopping/gpry_compas/fresh_confirmation_20260911'
sys.path.insert(0,str(ROOT.parent))
from cosmic_integration.lnl_surrogate.scheduled_audit import AuditPolicy,posterior_audit
from cosmic_integration.lnl_surrogate.stability_audit import stable_audit_acceptance

def main():
 status=json.loads((ROOT/'status.json').read_text());assert status['state']=='complete'
 data=(ROOT/'decisions.json').read_bytes();assert hashlib.sha256(data).hexdigest()==status['decision_sha256']
 decisions=json.loads(data);assert len(decisions)==48
 policy=AuditPolicy(sizes=(96,));stages=0
 for name,d in decisions.items():
  assert not d['reference_accessed'] and d['cohort']=='fresh_confirmation'
  accepted=None;cost=0
  for c in d['history']:
   f=ROOT/'training'/name/f'checkpoint_{c["training"]}.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==c['checkpoint_sha256']
   for s in c['stages']:
    assert s['size']==s['increment']==96
    with np.load(ROOT/'audits'/s['path']) as a:
     result=posterior_audit(a['predicted'],a['actual'],a['log_proposal'],a['groups'],202609122+c['training'],policy)
    assert result==s['result'];stages+=1
    if accepted is None:
     cost+=96
     if stable_audit_acceptance(result,c['diagnostics']['streaks']['joint']):accepted=dict(training=c['training'],audit=cost,total=c['training']+cost)
  assert accepted==d['rules']['fixed96'] and cost==d['audit_labels']['fixed96']
  ti,n,c=d['truth_index'],d['expected_events'],d['case'];prefix=f't{ti}_n{n}_'
  with np.load(ROOT/'training'/f'{prefix}perfect_c{c}_bo/catalogue.npz') as a,np.load(ROOT/'training'/name/'catalogue.npz') as b:
   for k in ['latent','rho','scales','y','bins','duration','truth']:np.testing.assert_array_equal(a[k],b[k])
 outcomes=json.loads((ROOT/'offline/outcomes.json').read_text());assert len(outcomes)==48 and {r['name'] for r in outcomes}==set(decisions)
 result=dict(cases=48,verified_audit_stages=stages,paired_catalogues=True,decision_sha256=status['decision_sha256'])
 (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
