"""Stage-one coverage and non-destructive parent continuation checks."""
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('gp4_campaign', Path(__file__).resolve().parents[1] / 'scripts/ozstar/gp4/campaign.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_stage_one_matches_pairs_and_excludes_reused_low_count():
    jobs = [dict(name=f'count_n{n}_{view}_c{c}_s17_{method}', group='count_history',
                 catalogue=f'{n}_{view}_{c}', strategy=method, seed=17)
            for c,n,view,method in itertools.product(range(200,205), (10,50,100,200,500,700,1000), ('perfect','uncertain'), ('bo','random'))]
    old = [j for j in jobs if any(j['name'].startswith(f'count_n{n}_') for n in (50,500)) and any(f'_c{c}_' in j['name'] for c in (200,201,202))]
    old += [dict(j, name=j['name'].replace('_s17_', '_s29_'), seed=29) for j in old if '_c200_' in j['name']]
    tasks = mod.build_tasks(jobs, [dict(kind='trajectory',job=j) for j in old])
    assert len(tasks) == len({t['job']['name'] for t in tasks}) == 92
    assert sum(t['parent'] for t in tasks) == 16
    assert sum(t['maximum'] == 1284 for t in tasks) == 68
    assert sum(t['group'] == 'high_count' for t in tasks) == 24
    pairs = {}
    for t in tasks:
        j=t['job']
        pairs.setdefault((j['catalogue'],j['seed'],t['maximum']),set()).add(j['strategy'])
    assert len(pairs)==46 and all(v=={'bo','random'} for v in pairs.values())
    assert not ({t['job']['name'] for t in tasks if not t['parent']} & {j['name'] for j in old})


def test_parent_copy_preserves_markers_and_rejects_tampering(tmp_path):
    source=tmp_path/'parent';source.mkdir()
    for name in ('checkpoint_1284.npz','timing_1284.json','complete.json','run.json'):
        (source/name).write_text(name)
    parent=dict(folder=str(source),files={p.name:mod.digest(p) for p in source.iterdir()})
    child=tmp_path/'child'
    mod.seed_parent(child,parent)
    mod.seed_parent(child,parent)
    assert sorted(p.name for p in child.iterdir())==['checkpoint_1284.npz','timing_1284.json']
    (child/'checkpoint_1284.npz').write_text('changed')
    with pytest.raises(ValueError,match='child checkpoint changed'):
        mod.seed_parent(child,parent)
    (source/'timing_1284.json').write_text('changed')
    with pytest.raises(ValueError,match='Checksum mismatch'):
        mod.seed_parent(tmp_path/'other',parent)


def test_parent_rejects_wrong_input_identity(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps(dict(job={'name':'x'},input_manifest_sha256='old',maximum=1284)))
    (tmp_path/'complete.json').write_text(json.dumps(dict(labels=1284)))
    with pytest.raises(ValueError,match='identity mismatch'):
        mod.parent_record(tmp_path,{'name':'x'},'new')


def test_remaining_matrix_excludes_prior_runs_and_preserves_predictive_pairs():
    jobs=[dict(name=f'count_n{n}_{view}_c{case}_s17_{method}',group='count_history')
          for n,view,case,method in itertools.product((10,50,100,200,500,700,1000),('perfect','uncertain'),range(200,205),('bo','random'))]
    jobs += [dict(name=f'predictive_n{n}_{view}_s17_bo',group='predictive') for n,view in itertools.product((30,1200),('perfect','uncertain'))]
    existing={j['name'] for j in jobs if j['group']=='count_history' and
              (int(j['name'].split('_')[1][1:])<=200 or
               ('count_n500_' in j['name'] and any(f'_c{c}_' in j['name'] for c in (200,201,202))) or
               ('count_n1000_' in j['name'] and any(f'_c{c}_' in j['name'] for c in (200,201))))}
    tasks=mod.build_remaining_tasks(jobs,existing)
    assert len(tasks)==44 and len(existing)==100
    assert len(existing | {t['job']['name'] for t in tasks})==144
    assert sum(t['group']=='predictive' for t in tasks)==4
    assert sum(t['maximum']==1284 for t in tasks)==2
