from types import SimpleNamespace

import pytest

from examples.simBenchmarks.Minari.run import evaluation_environment_seeds


def test_environment_seed_suites_and_old_checkpoint_default():
    assert evaluation_environment_seeds(SimpleNamespace(eval_episodes=2)) == [60000,60001]
    assert evaluation_environment_seeds(SimpleNamespace(eval_episodes=3,eval_seed_start=70000)) == [70000,70001,70002]
    for start,count in [(-1,1),(2**32-1,2),(0,0)]:
        with pytest.raises(ValueError,match='seed range'):
            evaluation_environment_seeds(SimpleNamespace(eval_episodes=count,eval_seed_start=start))
