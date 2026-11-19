"""Bounded-memory population identity audit; no full system arrays loaded."""
from pathlib import Path
import json
import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def main():
    paths=[REPO/f'tests/large_test_data/h5out_{n}M_reduced.h5' for n in [32,512]]
    result={'files':[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths]}
    with h5py.File(paths[0]) as small,h5py.File(paths[1]) as large:
        result['groups']={g:{name:{k:{'shape':list(ds.shape),'dtype':str(ds.dtype)} for k,ds in f[g].items()} for name,f in [('32M',small),('512M',large)]} for g in small}
        checks={}
        for group,fields in {'BSE_System_Parameters':['SEED','Mass@ZAMS(1)','Mass@ZAMS(2)','Metallicity@ZAMS(1)'],
                'BSE_Double_Compact_Objects':['SEED','Mass(1)','Mass(2)','Time','Coalescence_Time']}.items():
            for field in fields:
                a,b=small[group][field],large[group][field]
                mismatch=0
                for start in range(0,len(a),1000000):
                    end=min(start+1000000,len(a))
                    mismatch+=int(np.count_nonzero(a[start:end]!=b[start:end]))
                checks[group+'/'+field]={'prefix_mismatches':mismatch,'checked_rows':len(a)}
        result['prefix_checks']=checks
        result['seed_endpoints']={name:{g:[int(f[g]['SEED'][0]),int(f[g]['SEED'][-1])] for g in f} for name,f in [('32M',small),('512M',large)]}
    (HERE/'file_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
