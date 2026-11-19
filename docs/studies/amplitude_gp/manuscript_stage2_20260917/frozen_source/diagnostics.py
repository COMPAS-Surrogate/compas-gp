"""Post-training amplitude and held-out checks; never alter acquisition or GP fits."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp, ndtri
from scipy.spatial import cKDTree
from scipy.stats import gamma

import campaign as workflow


def density_summary(grid: np.ndarray, log_density: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Integrate a one-dimensional log density with explicit endpoint quadrature."""
    density = np.exp(log_density - np.max(log_density))
    density /= trapezoid(density, x=grid)
    cdf = np.r_[0., cumulative_trapezoid(density, x=grid)]
    mean = trapezoid(grid*density, x=grid)
    sd = np.sqrt(trapezoid((grid-mean)**2*density, x=grid))
    return dict(mean=float(mean), sd=float(sd), fractional_sd=float(sd/mean),
                quantiles=np.interp([.025,.16,.5,.84,.975],cdf,grid).tolist()), density, cdf


def conditional_metrics(amplitude: np.ndarray, observed: int, beta: float, predicted: np.ndarray) -> tuple[dict,dict]:
    """Compare GP slice with exact bounded Gamma; the observed count fixes its shape."""
    exact = observed*np.log(amplitude)-beta*amplitude
    direct, p, cdf = density_summary(amplitude, exact)
    gp, q, _ = density_summary(amplitude, predicted)
    shape = observed+1
    left,right=gamma.cdf(amplitude[[0,-1]],shape,scale=1/beta)
    if right-left < 1e-12:
        raise ValueError('Gamma interval needs tail-stable integration')
    analytic=(gamma.cdf(amplitude,shape,scale=1/beta)-left)/(right-left)
    cdf_error=float(np.max(abs(cdf-analytic)))
    if cdf_error > 1e-5:
        raise ValueError('Conditional grid is numerically unresolved')
    # Log normalization avoids tail underflow in directed KL.
    lp=exact-logsumexp(exact);lq=predicted-logsumexp(predicted)
    # Replace rectangular normalizers by trapezoidal normalizers.
    lp-=np.log(trapezoid(np.exp(lp),x=amplitude))
    lq-=np.log(trapezoid(np.exp(lq),x=amplitude))
    kl=[float(trapezoid(p*(lp-lq),x=amplitude)),float(trapezoid(q*(lq-lp),x=amplitude))]
    width=lambda s:s['quantiles'][3]-s['quantiles'][1]
    row=dict(observed=observed,beta=beta,gamma_shape=shape,direct=direct,gp=gp,
             analytic_cdf_max_error=cdf_error,directed_kl=kl,
             mean_error_sd=float(abs(gp['mean']-direct['mean'])/direct['sd']),
             width68_fraction_error=float(abs(width(gp)/width(direct)-1)),
             unbounded_fractional_sd=float(1/np.sqrt(shape)))
    return row,dict(amplitude=amplitude,direct_density=p,gp_density=q,exact=exact,predicted=predicted)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True)
    args=parser.parse_args();root=args.root.resolve()
    workflow.verify(root,workflow.read(root/'checksums.json'))
    manifest=workflow.read(root/'manifest.json');task=manifest['diagnostics'][args.index]
    original=Path(manifest['source_campaign']);inputs=original/'inputs'
    workflow.verify(original/'code',workflow.read(original/'code/paper_source_manifest.json'))
    workflow.verify(inputs,manifest['rate_files']);workflow.verify(inputs,manifest['reference_files'])
    sys.path.insert(0,str(original/'code/docs/studies/amplitude_gp'))
    import paper_campaign as campaign
    import jax
    import jax.numpy as jnp
    from cosmic_integration.lnl_surrogate.gpry_rules import latent_prediction,prepare_fantasies
    job=task['job'];folder=Path(task['folder']);labels=task['labels']
    checkpoint=folder/f'checkpoint_{labels}.npz';scorepath=folder/f'score_{labels}.json'
    if workflow.digest(checkpoint)!=task['checkpoint_sha256'] or workflow.digest(scorepath)!=task['score_sha256']:
        raise ValueError('Diagnostic source changed')
    catpath=inputs/'catalogues'/job['catalogue']
    if workflow.digest(catpath)!=job['catalogue_sha256']:raise ValueError('Catalogue changed')
    cat=dict(np.load(catpath));score=workflow.read(scorepath)
    state,training,_,_=campaign.restore(checkpoint,labels)
    out=root/'diagnostics'/job['name'];out.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,checksums_sha256=workflow.digest(root/'checksums.json'))
    if (out/'identity.json').exists() and workflow.read(out/'identity.json')!=identity:raise ValueError('Diagnostic identity changed')
    workflow.write(out/'identity.json',identity)
    engine=campaign.RateEngine(inputs)
    amplitude=np.linspace(float(campaign.LOW[2]),float(campaign.LOW[2]+campaign.WIDTH[2]),8193)
    mean=np.mean([s['direct']['mean'] for s in score['attempts'][-1]['scores']],axis=0)
    rows=[]
    for name,point in [('injection',cat['truth']),('reference_mean',mean)]:
        shape=np.array(point,copy=True);shape[2]=.012
        rates=engine(shape);duration=float(cat['duration']);number=len(cat['coeff'])
        beta=float(duration*rates.sum()/.012)
        points=np.tile(point,(len(amplitude),1));points[:,2]=amplitude
        unit=(points-campaign.LOW)/campaign.WIDTH
        predicted=campaign.study.pilot.base.predict(state,unit)
        # Independently check the complete implemented likelihood's slice shape.
        ix=np.linspace(0,len(amplitude)-1,9,dtype=int)
        implemented=campaign.evaluate(engine,cat,unit[ix])
        exact=number*np.log(amplitude[ix])-beta*amplitude[ix]
        np.testing.assert_allclose(implemented-implemented[4],exact-exact[4],atol=1e-8,rtol=1e-10)
        row,arrays=conditional_metrics(amplitude,number,beta,predicted)
        row.update(anchor=name,shape_point=np.asarray(point).tolist());rows.append(row)
        np.savez_compressed(out/f'amplitude_{name}.npz',**arrays)
    workflow.write(out/'amplitude.json',dict(job=job,labels=labels,rows=rows,
        scope='Conditional at two fixed shape points; joint marginal width is not required to obey counting law'))
    holdouts=[]
    if task['holdout']:
        fantasies=prepare_fantasies(state,jnp.zeros((1,4)),jnp.zeros(1))
        predictor=jax.jit(jax.vmap(latent_prediction,in_axes=(None,0,None)))
        levels=np.array([.5,.68,.9,.95,.99])
        for scramble in (0,1):
            path=inputs/'references'/f"t{job['truth_index']}_n{job['reference_count']}_{scramble}.npz"
            with np.load(path) as f:
                indices=np.random.default_rng(202609170+scramble).choice(len(f['phi']),2048,replace=False)
                phi,rates,shapeq=f['phi'][indices],f['rates'][indices],f['logq'][indices]
            number=len(cat['coeff']);duration=float(cat['duration'])
            beta=duration*rates.sum(axis=1)/.012
            amplitude,aq=campaign.study.amplitude_design(beta,number,16,202609180+scramble)
            repeated=np.repeat(phi,16,axis=0)
            points=np.column_stack([repeated[:,:2],amplitude.ravel(),repeated[:,2]])
            unit=(points-campaign.LOW)/campaign.WIDTH
            distance=cKDTree(training).query(unit)[0].min()
            if distance<=1e-10:raise ValueError('Holdout overlaps training')
            constant=np.concatenate([np.log(duration*(r@cat['coeff'].T)).sum(axis=1) for r in np.array_split(rates,8)])
            exact=(constant[:,None]+number*np.log(amplitude/.012)-beta[:,None]*amplitude).ravel()
            logq=np.repeat(shapeq,16)+aq.ravel()
            pairs=[predictor(state,jnp.asarray(part),fantasies) for part in np.split(unit,len(unit)//128)]
            mu=np.concatenate([np.asarray(p[0]) for p in pairs]);variance=np.concatenate([np.asarray(p[1]) for p in pairs])
            reference,scale=float(state['reference']),float(state['scale'])
            predicted=reference-(scale*np.maximum(mu,0))**2
            np.testing.assert_allclose(predicted,campaign.study.pilot.base.predict(state,unit),atol=1e-7,rtol=0)
            weights=np.exp(exact-logq-logsumexp(exact-logq));residual=predicted-exact
            centered=residual-weights@residual;coverage=[]
            for level in levels:
                z=ndtri((1+level)/2)
                low=reference-(scale*np.maximum(mu+z*np.sqrt(variance),0))**2
                high=reference-(scale*np.maximum(mu-z*np.sqrt(variance),0))**2
                coverage.append(float(weights@((exact>=low)&(exact<=high))))
            holdouts.append(dict(scramble=scramble,points=len(unit),raw_rmse=float(np.sqrt(weights@residual**2)),
                centered_rmse=float(np.sqrt(weights@centered**2)),ess=float(1/(weights@weights)),
                max_weight=float(weights.max()),levels=levels.tolist(),coverage=coverage,
                minimum_training_distance=float(distance),reference_sha256=workflow.digest(path)))
            np.savez_compressed(out/f'holdout_{scramble}.npz',indices=indices,points=points,exact=exact,predicted=predicted,
                                logq=logq,weights=weights,latent_mean=mu,latent_variance=variance)
        workflow.write(out/'holdout.json',dict(job=job,labels=labels,rows=holdouts,
            scope='Withheld from training; subsamples of existing independent numerical reference designs. No calibration tuning. Importance ESS does not replace between-design precision checks.'))
    workflow.write(out/'status.json',dict(state='complete',amplitude_slices=len(rows),holdout_designs=len(holdouts)))


if __name__=='__main__':main()
