"""Publish the supplemental random-acquisition comparison after all runs finish."""
import json
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    root=HERE/'random_results'
    if not (root/'runtime.json').exists():raise RuntimeError('Random comparison is incomplete')
    rows=json.loads((root/'records.json').read_text());assert len(rows)==200
    summary={}
    for n in [30,100,300,1000]:
        group=[r for r in rows if r['expected_events']==n and r['points']==738];assert len(group)==10
        summary[n]={'validated':sum(r['validated_pass'] for r in group),'numerical_failures':sum(not r['numerical_pass'] for r in group),'median_forward_kl':float(np.median([max(m['kl_direct_to_gp'] for m in r['metrics']) for r in group]))}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    counts=', '.join(str(summary[n]['validated']) for n in [30,100,300,1000])
    failures=sum(v['numerical_failures'] for v in summary.values())
    text=(r'As a supplementary control, we replace adaptive acquisition by independent uniform prior draws after the common initialization. At 738 training points, the numbers of catalogues satisfying the full validation criteria are '+counts+r' out of ten, for 30, 100, 300 and 1000 expected detections, respectively. This comparison uses one random design shared across catalogues. It does not establish performance averaged over random-design seeds. ')
    if failures:text+=f'{failures} final random-design assessments remain numerically unresolved and are not counted as successes. '
    (HERE.parents[2]/'overleaf/output/bo_study/random_results.tex').write_text(text+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
