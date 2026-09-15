"""Exercise job argument handling without Slurm or COMPAS population data."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script,random', [('slurm.sh', False), ('slurm_32M.sh', False),
                                        ('random/slurm.sh', True)])
def test_jobs_quote_paths_and_refuse_existing_output(tmp_path, script, random):
    envdir = tmp_path/'env space'
    (envdir/'bin').mkdir(parents=True)
    python = envdir/'bin/python'
    python.write_text('#!/bin/sh\nexit 0\n')
    python.chmod(0o755)
    capture = tmp_path/'arguments.json'
    for name in ('run_cosmic_integration', 'generate_random_samples'):
        cli = envdir/'bin'/name
        cli.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                       'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
        cli.chmod(0o755)
    source = tmp_path/'input space.h5'
    source.touch()
    output = tmp_path/'output space'/'result'
    env = dict(os.environ, COMPAS_REPO=str(ROOT), COMPAS_ENV=str(envdir),
               COMPAS_INPUT=str(source), COMPAS_OUTPUT=str(output), CAPTURE=str(capture),
               SLURM_ARRAY_TASK_ID='2', SLURM_CPUS_PER_TASK='1', TMPDIR=str(tmp_path))
    path = ROOT/'docs/studies/generate_data'/script
    subprocess.run(['bash', str(path)], env=env, check=True, capture_output=True)
    args = json.loads(capture.read_text())
    actual_output = Path(str(output)+'_2.csv') if random else output
    assert str(actual_output) in args
    assert (str(source) if random else source.name) in args
    (actual_output if random else Path(str(actual_output)+'.csv')).touch()
    result = subprocess.run(['bash', str(path)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Output already exists' in result.stderr


def test_setup_refuses_existing_environment(tmp_path):
    env = dict(os.environ, COMPAS_REPO=str(ROOT), COMPAS_ENV=str(tmp_path))
    result = subprocess.run(['bash', str(ROOT/'scripts/ozstar/setup.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Refusing to modify' in result.stderr


def test_cosmic_integration_console_entry_is_callable():
    import importlib
    import tomllib
    config = tomllib.loads((ROOT/'pyproject.toml').read_text())
    module, function = config['project']['scripts']['run_cosmic_integration'].split(':')
    assert callable(getattr(importlib.import_module(module), function))
