import runpy
from pathlib import Path
import pytest

compare=runpy.run_path(str(Path(__file__).parent/'benchmarks/rollflow_acceptance.py'))['compare']


def metrics(instant,average):
    return {f'{mode}_k{k}':{'mse':value} for mode,value in [('instant',instant),('average',average)] for k in (1,2,4)}


def test_untrained_average_branch_is_not_the_reference():
    results=compare([metrics(.02,5.)],[metrics(.02,.10)])
    assert results['average_k1_vs_instant_k1']['ratio']==pytest.approx(5.)
    assert not results['average_k1_vs_instant_k1']['mean_noninferior']


def test_mean_improvement_is_not_all_seed_improvement():
    result=compare([metrics(.1,1.),metrics(.1,1.)],[metrics(.1,.01),metrics(.1,.15)])
    assert result['average_k2_vs_instant_k4']['mean_noninferior']
    assert not result['average_k2_vs_instant_k4']['all_seeds_noninferior']


def test_incomplete_or_nonfinite_pairs_are_rejected():
    with pytest.raises(ValueError):compare([],[])
    with pytest.raises(ValueError):compare([metrics(.1,.1)],[])
    with pytest.raises(ValueError):compare([metrics(.1,.1)],[metrics(.1,float('nan'))])
