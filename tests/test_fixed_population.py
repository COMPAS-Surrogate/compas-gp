"""Contraction parity on a small population, including delay/grid boundaries."""
from types import SimpleNamespace
import numpy as np
from cosmic_integration.ratesSampler.fixed_population import FixedPopulationRates
from cosmic_integration.ratesSampler.ratesSampler import CosmicIntegration
from cosmic_integration.ratesSampler.grid_diagnostics import system_grid_moments


def test_fixed_population_matches_legacy_delay_lookup():
    ci = SimpleNamespace()
    ci.compas = SimpleNamespace(mass1=np.array([2.,8.,20.,30.]),mass2=np.array([2.,4.,10.,20.]),
        Zsystems=np.array([.002,.01,.002,.01]),delayTime=np.array([0.,1.,4.,20.]),
        zamsZmin=.001,zamsZmax=.02,massEvolvedPerBinary=2.,nSystems=50)
    ci.redshifts=np.arange(5.)
    ci.times=np.array([10.,7.,5.,3.,1.])
    ci.nRedshiftsDetection=3
    ci.distances=np.ones(5)
    ci.shellVolumes=np.arange(1.,6.)
    ci.SE=SimpleNamespace(SNRgridAt1Mpc=None,detectionProbabilityFromSNR=None)
    ci.CalculateRedshiftRelatedParams=lambda: None
    ci.FindDetectionProbability=lambda *args: np.full((4,3),.3)
    ci.CalculateSFR=lambda z,a,d: a*(1+z)**d
    def density(z,*args,p_Alpha=-.3,p_Sigma=.2):
        return np.exp(p_Alpha*z[:,None])*np.array([[p_Sigma,1.,2.]]),np.array([.001,.005,.02]),.5
    ci.FindZdistribution=density
    edges=np.array([0.,5.,np.inf]); groups=np.array([0,1,3])
    fast=FixedPopulationRates(ci,edges,groups)
    for alpha,sigma,a,d in [(-.3,.2,.01,2.),(-.1,.5,.02,1.)]:
        pdf,metals,pdraw=density(ci.redshifts,p_Alpha=alpha,p_Sigma=sigma)
        formed=ci.CalculateSFR(ci.redshifts,a,d)/100
        _,merged=CosmicIntegration.FindFormationAndMergerRates(ci,4,ci.redshifts,ci.times,formed,pdf,metals,pdraw,ci.compas.Zsystems,ci.compas.delayTime)
        rate=merged[:,:3]*.3*ci.shellVolumes[:3]/(1+ci.redshifts[:3])
        masses=(ci.compas.mass1*ci.compas.mass2)**.6/(ci.compas.mass1+ci.compas.mass2)**.2
        expected,_=system_grid_moments(rate,masses,edges,groups)
        np.testing.assert_allclose(fast(alpha,sigma,a,d),expected,rtol=1e-14,atol=1e-15)
