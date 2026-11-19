#!/usr/bin/env bash
# Run on OzSTAR after freezing a campaign with campaign.py prepare.
set -euo pipefail
: "${COMPAS_CAMPAIGN:?Set COMPAS_CAMPAIGN}"
: "${COMPAS_REPO:?Set COMPAS_REPO}"
: "${COMPAS_ENV:?Set COMPAS_ENV}"
export COMPAS_CAMPAIGN COMPAS_REPO COMPAS_ENV
cd "$COMPAS_CAMPAIGN"
# Atomic reservation prevents duplicate submissions, including concurrent invocations.
mkdir submission.lock
python3 - <<'PY'
import hashlib,json
from pathlib import Path
for name,expected in json.loads(Path('checksums.json').read_text()).items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Source checksum mismatch: '+name)
PY
mkdir -p logs
count=$(python3 -c 'import json; print(len(json.load(open("manifest.json"))["tasks"]))')
last=$((count - 1))
COMPAS_STAGE=pilot
export COMPAS_STAGE
pilot=$(sbatch --parsable --export=ALL --output="$COMPAS_CAMPAIGN/logs/pilot-%j.out" --error="$COMPAS_CAMPAIGN/logs/pilot-%j.err" run.slurm)
printf '%s\n' "$pilot" > pilot_job_id.txt
COMPAS_STAGE=run
export COMPAS_STAGE
array=$(sbatch --parsable --export=ALL --dependency="afterok:${pilot%%;*}" --kill-on-invalid-dep=yes --array="0-$last%4" --output="$COMPAS_CAMPAIGN/logs/run-%A_%a.out" --error="$COMPAS_CAMPAIGN/logs/run-%A_%a.err" run.slurm)
printf '%s\n' "$array" > array_job_id.txt
python3 - <<'PY'
import datetime,json
from pathlib import Path
record=dict(submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pilot=Path('pilot_job_id.txt').read_text().strip(),array=Path('array_job_id.txt').read_text().strip(),
            tasks=len(json.loads(Path('manifest.json').read_text())['tasks']),concurrency=4,scope=json.loads(Path('manifest.json').read_text())['scope'])
Path('submission.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY

# Diagnostics use existing completed states and need no new production training.
if python3 -c 'import json,sys; sys.exit(not bool(json.load(open("manifest.json")).get("diagnostics")))'; then
    COMPAS_STAGE=diagnostics
    export COMPAS_STAGE
    diagnostic=$(sbatch --parsable --export=ALL --dependency="afterok:${pilot%%;*}" --kill-on-invalid-dep=yes --array=0-11%2 --output="$COMPAS_CAMPAIGN/logs/diagnostics-%A_%a.out" --error="$COMPAS_CAMPAIGN/logs/diagnostics-%A_%a.err" run.slurm)
    printf '%s\n' "$diagnostic" > diagnostics_job_id.txt
    python3 - <<'PYDIAG'
import json
from pathlib import Path
p=Path('submission.json');d=json.loads(p.read_text())
d['diagnostics']=Path('diagnostics_job_id.txt').read_text().strip();d['diagnostic_tasks']=12
p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
PYDIAG
fi
