import json,hashlib
from pathlib import Path
import numpy as np
root=Path('docs/studies/amplitude_gp')
rows=[]
for p in sorted((root/'continued_bo_20260914').glob('*/complete.json')):
 folder=p.parent; meta=json.loads(p.read_text())
 cp=folder/'checkpoint_1494.npz'; f=np.load(cp); x=f['unit']; y=f['lnl']
 prior=root/'small_batch_20260914'/folder.name/'checkpoint_744.npz'; old=np.load(prior)
 assert hashlib.sha256(cp.read_bytes()).hexdigest()==meta['checkpoint_sha256']
 assert hashlib.sha256(prior.read_bytes()).hexdigest()==meta['initial_sha256']
 assert np.array_equal(x[:744],old['unit']) and np.array_equal(y[:744],old['lnl'])
 assert x.shape==(1494,4) and len(np.unique(x,axis=0))==1494
 assert np.isfinite(x).all() and np.isfinite(y).all() and np.all((x>=0)&(x<=1))
 batches=json.loads((folder/'batches.json').read_text()); assert len(batches)==250
 predictions=[]; actual=[]; diagnostics=[]
 for i,b in enumerate(batches):
  k=744+3*i; assert b['before_labels']==k
  assert np.array_equal(x[k:k+3],b['unit']) and np.array_equal(y[k:k+3],b['actual'])
  assert np.min(np.sum((x[k:k+3,None,:]-x[None,:k,:])**2,axis=-1))>1e-8
  assert np.isfinite(b['prediction']).all()
  predictions.extend(b['prediction']); actual.extend(b['actual']); diagnostics.extend(b['acquisition'])
 pred=np.array(predictions); act=np.array(actual)
 def error(a,b):
  near=b>y.max()-2
  return dict(count=int(near.sum()),rmse=float(np.sqrt(np.mean((a[near]-b[near])**2))))
 rows.append(dict(job=folder.name,checks_passed=True,acquired=750,
  first90_peak=error(pred[:90],act[:90]),last90_peak=error(pred[-90:],act[-90:]),
  optimizer_origin_fraction=float(np.mean([d['origin']=='optimizer' for d in diagnostics])),
  optimizer_success_fraction=sum(d['optimizer_successes'] for d in diagnostics)/sum(d['starts'] for d in diagnostics)))
result=dict(scope='Continuation provenance, all-coordinate bounds, exact prefix preservation, batch alignment and novelty. Pre-acquisition errors use selected points within 2 lnL of the final best training label, not independent validation.',jobs=rows,passed=True,total_new_labels=sum(r['acquired'] for r in rows))
out=root/'continued_bo_20260914'/'acquisition_audit.json'; out.write_text(json.dumps(result,indent=2)+'\n')
for r in rows: print(r['job'],round(r['first90_peak']['rmse'],3),round(r['last90_peak']['rmse'],3),round(r['optimizer_success_fraction'],3))
