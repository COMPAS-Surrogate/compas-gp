"""Detected-rate maps at contrasting cosmic-history parameters; no new inference."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.sparse import load_npz
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'event_count_bo'))
from run_study import configurations, BinnedCosmicIntegrator, FixedPopulationRates
from publish_figures import plt, save


def main():
    ci=BinnedCosmicIntegrator.from_compas_fpath(str(REPO/'tests/large_test_data/h5out_32M_reduced.h5'))
    mass,z,_=configurations()['mass10'];fast=FixedPopulationRates(ci,mass,z)
    block=HERE.parent/'population_stability/block_results'
    meta=json.loads((block/'population_meta.json').read_text())
    ci.compas.nSystems=512000000
    ci.compas.zamsZmin=meta['zmin'];ci.compas.zamsZmax=meta['zmax']
    fast.p_draw=1/np.log(meta['zmax']/meta['zmin'])
    fast.operator=sum((load_npz(block/f'block_{i:02d}.npz') for i in range(1,16)),load_npz(block/'block_00.npz'))
    fid=np.array([-.3,.35,.01,4.7])
    # Columns alpha, sigma, d; rows low/high. Amplitude changes normalization only.
    parameters=[]
    for row in range(2):
        for j,values in [(0,[-.45,-.15]),(1,[.15,.55]),(3,[4.3,5.1])]:
            p=fid.copy();p[j]=values[row];parameters.append(p)
    rates=np.array([fast(*p) for p in parameters]);ze=z*.1
    finite=np.isfinite(mass[:-1]) & np.isfinite(mass[1:])
    positive=mass[:-1]>0
    keep=finite & positive
    assert np.all(rates[:,~keep,:]==0), 'Cannot omit populated zero-mass or infinite guard bins'
    idx=np.flatnonzero(keep);assert np.all(np.diff(idx)==1)
    edges=mass[idx[0]:idx[-1]+2]
    area=np.diff(edges)[:,None]*np.diff(ze)[None,:]
    density=rates[:,keep,:]/area
    assert np.allclose((density*area).sum(axis=(1,2)),rates.sum(axis=(1,2)))
    # Every block contributes with the same normalization for 512M systems.
    fid_full=fast(*fid)
    full_operator=fast.operator
    block_sum=np.zeros_like(fid_full)
    for i in range(16):
        fast.operator=load_npz(block/f'block_{i:02d}.npz')
        block_sum+=fast(*fid)
    fast.operator=full_operator
    assert np.allclose(fid_full,block_sum,rtol=1e-12)
    low=fast(fid[0],fid[1],.005,fid[3]);high=fast(fid[0],fid[1],.015,fid[3])
    assert np.allclose(high,3*low,rtol=1e-12)
    vmax=10**np.ceil(np.log10(density.max()));vmin=vmax/1e5
    norm=LogNorm(vmin=vmin,vmax=vmax)
    cmap=plt.get_cmap('inferno').copy();cmap.set_bad('#eeeeee');cmap.set_under('#160b39')
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(cmap(np.linspace(.08,1,256)))
    cmap.set_bad('#eeeeee');cmap.set_under(cmap(0))
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.5),sharex=True,sharey=True,layout='constrained')
    for i,ax in enumerate(axes.flat):
        im=ax.pcolormesh(ze,edges,np.ma.masked_less_equal(density[i],0),cmap=cmap,norm=norm,rasterized=True,shading='flat')
        ax.set(yscale='log',xlim=(ze[0],ze[-1]),ylim=(edges[0],edges[-1]))
        ax.tick_params(which='both',color='.6',labelcolor='black')
        for spine in ax.spines.values(): spine.set_color('.6')
        ax.set_yticks([2,5,10,20,40,80]);ax.yaxis.set_major_formatter(ScalarFormatter())
        if i>=3:ax.set_xlabel('Redshift')
        if i%3==0:ax.set_ylabel(r'Chirp mass [$M_\odot$]')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='#eeeeee',edgecolor='.7',label='Zero rate')],loc='outside upper center',fontsize=8)
    cb=fig.colorbar(im,ax=axes,extend='min',pad=.02,fraction=.04)
    cb.set_label(r'$\mathrm{d}^2 R_{\mathrm{det}}/(\mathrm{d}z\,\mathrm{d}\mathcal{M}_c)$ [$\mathrm{yr}^{-1}\,M_\odot^{-1}$]')
    save(fig,'rate_heatmaps')
    np.savez(HERE/'rate_maps.npz',parameters=parameters,mass_edges=mass,z_edges=ze,rates=rates)
    info={'parameters':np.array(parameters).tolist(),'parameter_order':['alpha','sigma','a_SFR','d_SFR'],'total_detected_rates_per_year':rates.sum(axis=(1,2)).tolist(),'colour_limits':[vmin,vmax],'scope':'Fixed blocks0-15, O3 approximation, existing mass10 grid; exact bin-average detected rate density per linear mass and redshift; logarithmic mass display only','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
