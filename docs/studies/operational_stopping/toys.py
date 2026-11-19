"""Toy log densities exposed to the learner only through point evaluations."""
import numpy as np
from scipy.special import logsumexp

FAMILIES = ['correlated', 'curved', 'two_modes', 'narrow', 'hidden_mode']


def make_target(family: str, seed: int):
    """Return a deterministic density on the unchanged unit-square prior."""
    rng = np.random.default_rng(seed)
    centre = rng.uniform(.4,.6,2)
    angle = rng.uniform(-.6,.6)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    def gaussian(x, mean, widths):
        z = (np.asarray(x)-mean)@rotation/widths
        return -.5*np.sum(z*z,axis=1)-np.log(2*np.pi*np.prod(widths))

    if family == 'correlated':
        return lambda x: gaussian(x, centre, [.045,.16])
    if family == 'narrow':
        return lambda x: gaussian(x, centre, [.022,.035])
    if family == 'curved':
        def curved(x):
            u=np.asarray(x)[:,0]-centre[0]
            v=np.asarray(x)[:,1]-centre[1]-2.5*u*u
            return -.5*((u/.15)**2+(v/.04)**2)-np.log(2*np.pi*.15*.04)
        return curved
    if family in ['two_modes','hidden_mode']:
        a=np.array([.3,.35])+rng.uniform(-.035,.035,2)
        b=np.array([.73,.72])+rng.uniform(-.035,.035,2)
        wa,wb=([.065,.08],[.065,.08]) if family=='two_modes' else ([.08,.09],[.007,.007])
        mixture=.5 if family=='two_modes' else .7
        return lambda x: logsumexp(np.vstack([np.log(mixture)+gaussian(x,a,wa),
                           np.log1p(-mixture)+gaussian(x,b,wb)]),axis=0)
    raise ValueError(f'Unknown family: {family}')
