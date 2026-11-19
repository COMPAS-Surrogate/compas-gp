"""Score the additional 100-label fits with the campaign's cached direct designs."""
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
import count_history_manuscript as p


def main():
    for n in (100,1000):
        for mode in ('perfect','uncertain'):
            name=f't0_n{n}_{mode}_c200_bo';folder=p.ROOT/'offline'/name
            out=p.OUT/'early100'/name
            provenance=json.loads((folder/'reference.json').read_text())
            hashes={};refs=p.old.validation.score.cached_references(0,provenance['proposal_expected_events'],hashes)
            assert all(provenance['source_sha256'][k]==v for k,v in hashes.items())
            with np.load(p.ROOT/'training'/name/'catalogue.npz') as d:cat=dict(d)
            checked=p.old.validation.score.direct_checks(refs,cat,cat['truth'])
            cp=out/'checkpoint_100.npz'
            row=p.old.validation.score_checkpoint(cp,refs,cat,cat['truth'],checked,
                dict(name=name,truth_index=0,expected_events=n,mode=mode,case=200,strategy='bo',points=100))
            assert all(row[k] for k in ('reference_numerical','gp_numerical','four_numerical'))
            state,_,_=p.old.BASE.load_checkpoint(cp)
            ws=[p.old.BASE.predict(state,(x-p.old.BASE.LOW)/p.old.BASE.WIDTH)-q for x,_,q in refs]
            np.savez_compressed(out/'posterior_100.npz',**{f'logweights_{i}':w-logsumexp(w) for i,w in enumerate(ws)})
            with np.load(folder/'reference_weights.npz') as d:
                kls=[p.old.validation.score.directed_kls(w,d[f'late_gp_{i}']) for i,w in enumerate(ws)]
            row.update(symmetric_kl_to_late_gp=float(np.mean(kls)),late_reference_accepted=True,
                       late_reference_is_self=False,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
            (out/'score.json').write_text(json.dumps(row,indent=2)+'\n')
            print(name,'numerically resolved; direct KL',row['symmetric_kl'],flush=True)


if __name__=='__main__':main()
