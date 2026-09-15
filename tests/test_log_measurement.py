import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import logsumexp
from cosmic_integration.observation.log_measurement import bin_coefficients,log_marginal_events,mock_pe_samples


def test_exact_bin_average_resolves_narrow_measurement():
    y=np.log([[2.03,.22]]);s=np.array([[.002,.08]])
    lo=np.array([[1.,.0],[2.,.2]]);hi=np.array([[2.,.2],[3.,.4]])
    actual=bin_coefficients(y,s,lo,hi)[0]
    expected=[]
    for l,h in zip(lo,hi):
        v=[]
        for j in range(2):
            center=np.exp(y[0,j]);points=[center] if l[j]<center<h[j] else []
            val=quad(lambda x:norm.pdf(y[0,j],loc=np.log(x),scale=s[0,j]),l[j],h[j],points=points,epsabs=1e-10)[0]/(h[j]-l[j]);v.append(val)
        expected.append(np.prod(v))
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-12)


def test_event_integrals_do_not_have_to_sum_to_total_rate():
    e=np.array([[10.,30.,10.],[.1,.3,.1]]);total=np.array([2.,.02]);T=.5
    value=log_marginal_events(e,total,T)
    for i in range(2):
        f=lambda a:np.exp(-T*total[i]*a/.012)*np.prod(T*e[i]*a/.012)/.01
        expected=np.log(quad(f,.005,.015,epsabs=1e-13)[0])
        np.testing.assert_allclose(value[i],expected,rtol=1e-10)


def test_mock_pe_prior_cancels_with_correct_coordinate_measure():
    y=np.log([2.,.4]);s=np.array([.1,.3]);lo=np.array([.5,0.]);hi=np.array([5.,1.5])
    direct=bin_coefficients(y[None,:],s[None,:],lo[None,:],hi[None,:])[0,0]*np.prod(hi-lo)
    for powers in [np.array([0.,0.]),np.array([1.,2.])]:
        samples,prior,Z=mock_pe_samples(y,s,lo,hi,powers,100000,np.random.default_rng(12))
        assert np.all((samples>=lo)&(samples<=hi))
        np.testing.assert_allclose(Z*np.mean(1/prior),direct,rtol=.015)


def test_event_amplitude_extreme_tail_stays_finite():
    e=np.array([[1e5,3e5,1e5]]);total=np.array([1e5]);T=.5
    beta=T*total[0]/.012;lower=.005;n=e.shape[1]
    # Independent boundary-scaled quadrature; beyond u=200 the omitted mass is negligible.
    integral=quad(lambda u:(1+u/(beta*lower))**n*np.exp(-u),0,200,epsabs=1e-13)[0]
    expected=np.log(T*e[0]/.012).sum()+n*np.log(lower)-beta*lower-np.log(beta*.01)+np.log(integral)
    np.testing.assert_allclose(log_marginal_events(e,total,T)[0],expected,rtol=1e-12)
