import h5py
import numpy as np
from cosmic_integration.ratesSampler.population_stream import load_selected_population


def test_stream_selection_and_seed_join(tmp_path):
    path=tmp_path/'population.h5'
    with h5py.File(path,'w') as f:
        g=f.create_group('BSE_System_Parameters')
        g['SEED']=[40,10,30,20,50]
        g['Metallicity@ZAMS(1)']=[.04,.01,.03,.02,.05]
        g=f.create_group('BSE_Double_Compact_Objects')
        for key,values in {'SEED':[10,20,30,40],'Mass(1)':[1.,2.,3.,4.],
            'Mass(2)':[1.,1.,1.,1.],'Time':[1.,2.,3.,4.],
            'Coalescence_Time':[10.,20.,30.,40.],'Merges_Hubble_Time':[1,1,0,1]}.items():g[key]=values
        g=f.create_group('BSE_Common_Envelopes')
        g['SEED']=[20,50,20]
        g['Immediate_RLOF>CE']=[1,0,0]
        g['Optimistic_CE']=[0,1,0]
    pop=load_selected_population(path,123.,chunk_size=2)
    np.testing.assert_array_equal(pop.dcoSeeds,[10,40])
    np.testing.assert_array_equal(pop.system_rows,[1,0])
    np.testing.assert_array_equal(pop.Zsystems,[.01,.04])
    np.testing.assert_array_equal(pop.delayTime,[11.,44.])
    assert pop.nSystems==5 and pop.zamsZmin==.01 and pop.zamsZmax==.05
    assert pop.massEvolvedPerBinary==123.
    assert not pop.legacy_metallicity_order_matches
