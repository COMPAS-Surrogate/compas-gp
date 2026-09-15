"""Record the CPU/x64 numerical environment and verify locked core versions."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import tomllib

import jax
import jax.numpy as jnp

root = Path(__file__).resolve().parents[2]
lock = tomllib.loads((root/'uv.lock').read_text())
names = ('jax', 'jaxlib', 'gpjax', 'equinox', 'optax', 'numpyro')
versions = {name: metadata.version(name) for name in names}
for name, version in versions.items():
    assert version in {p['version'] for p in lock['package'] if p['name'] == name}, (name, version)
assert sys.version_info[:2] == (3, 12)
assert jax.config.x64_enabled
assert jax.default_backend() == 'cpu'
x = jnp.array([[2., .3], [.3, 1.]], dtype=jnp.float64)
assert float(jnp.max(jnp.abs(x @ jnp.linalg.solve(x, jnp.eye(2))-jnp.eye(2)))) < 1e-12
print(json.dumps(dict(python=sys.version, platform=platform.platform(), versions=versions,
    x64=jax.config.x64_enabled, devices=[str(d) for d in jax.devices()],
    lock_sha256=hashlib.sha256((root/'uv.lock').read_bytes()).hexdigest()), indent=2))
