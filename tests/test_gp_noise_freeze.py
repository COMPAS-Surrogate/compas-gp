import numpy as np
import paramax
from cosmic_integration.lnl_surrogate.jax_active_learner import JaxGPConfig,JaxTrainingData,fit_exact_gp


def test_noise_remains_fixed_through_optimizer():
    x=np.linspace(0,1,16)[:,None]
    y=(np.sin(8*x)+.3*np.where(np.arange(16)[:,None]%2,1.,-1.))
    data=JaxTrainingData(x,y)
    for optimize in [False,True]:
        fitted=fit_exact_gp(data,np.array([[0.],[1.]]),config=JaxGPConfig(
            initial_noise_stddev=.003,optimise_noise=optimize,optimisation_steps=40),seed=1)
        noise=float(paramax.unwrap(fitted.posterior).likelihood.obs_stddev)
        if optimize:assert abs(noise-.003)>1e-5
        else:np.testing.assert_allclose(noise,.003,rtol=1e-13,atol=0.)
        mean,var=fitted.predict_f(x)
        assert np.isfinite(mean).all() and np.isfinite(var).all()
