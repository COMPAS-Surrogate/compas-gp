"""Bounded-memory extraction for ALL, Hubble-merging, pessimistic/no-RLOF DCOs.

This deliberately supports only the validation study's fixed selection. It joins
metallicity by SEED and checks uniqueness instead of assuming HDF5 row alignment.
Normalization and selection-effect objects must come from the matched integrator.
"""
from pathlib import Path
from types import SimpleNamespace
import logging
import h5py
import numpy as np

LOG = logging.getLogger(__name__)


def load_selected_population(path: str | Path, mass_evolved_per_binary: float,
                             chunk_size: int = 1000000) -> SimpleNamespace:
    """Stream large system/CE groups; retain only selected DCO properties."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    with h5py.File(path, 'r') as f:
        dco=f['BSE_Double_Compact_Objects']
        seeds=dco['SEED'][...]
        order=np.argsort(seeds)
        sorted_seeds=seeds[order]
        if np.any(np.diff(sorted_seeds)==0):
            raise ValueError('DCO seeds are not unique')
        mask=dco['Merges_Hubble_Time'][...].astype(bool)
        ce=f['BSE_Common_Envelopes']
        for start in range(0,len(ce['SEED']),chunk_size):
            sl=slice(start,start+chunk_size)
            bad=ce['Immediate_RLOF>CE'][sl].astype(bool)|ce['Optimistic_CE'][sl].astype(bool)
            bad_seeds=ce['SEED'][sl][bad]
            index=np.searchsorted(sorted_seeds,bad_seeds)
            valid=index<len(seeds)
            index=index[valid]; bad_seeds=bad_seeds[valid]
            matching=index[sorted_seeds[index]==bad_seeds]
            mask[order[matching]]=False
        selected=seeds[mask]
        LOG.info('%s: %d selected DCOs',Path(path).name,len(selected))
        order=np.argsort(selected); sorted_seeds=selected[order]
        system_rows=np.empty(len(selected), dtype=np.int64)
        metallicity=np.empty(len(selected)); found=np.zeros(len(selected),dtype=int)
        system_order=[]
        sys=f['BSE_System_Parameters']
        zmin,zmax=np.inf,-np.inf
        for start in range(0,len(sys['SEED']),chunk_size):
            sl=slice(start,start+chunk_size)
            ss=sys['SEED'][sl]; zz=sys['Metallicity@ZAMS(1)'][sl]
            zmin=min(zmin,zz.min());zmax=max(zmax,zz.max())
            index=np.searchsorted(sorted_seeds,ss)
            valid=index<len(selected)
            rows=np.flatnonzero(valid)
            index=index[valid]
            matching=sorted_seeds[index]==ss[rows]
            positions=order[index[matching]]
            metallicity[positions]=zz[rows[matching]]
            system_rows[positions]=start+rows[matching]
            np.add.at(found,positions,1)
            system_order.extend(positions)
            if (start//chunk_size+1)%64==0:
                LOG.info('Joined %d million system rows',(start+chunk_size)//1000000)
        if np.any(found!=1):
            raise ValueError('Selected DCOs must each match exactly one system record')
        return SimpleNamespace(dcoSeeds=selected,system_rows=system_rows,mass1=dco['Mass(1)'][...][mask],
            mass2=dco['Mass(2)'][...][mask],
            delayTime=dco['Time'][...][mask]+dco['Coalescence_Time'][...][mask],
            Zsystems=metallicity,zamsZmin=float(zmin),zamsZmax=float(zmax),
            nSystems=len(sys['SEED']),massEvolvedPerBinary=mass_evolved_per_binary,
            legacy_metallicity_order_matches=bool(np.array_equal(system_order,np.arange(len(selected)))))
