import copy
import importlib.util
from pathlib import Path

import pytest
import torch


path = Path(__file__).resolve().parents[1] / 'examples/simBenchmarks/Minari/train_files/branch_fm.py'
spec = importlib.util.spec_from_file_location('minari_fm_branch', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload():
    return {'arguments': {'variant': 'rolling_lsd', 'fm_warmup': 4, 'w_lsd': 2., 'p_fm': .3},
            'head_config': {'w_lsd': 2., 'p_fm': .3, 'velocity_mode': 'average'}, 'step': 4,
            'model': {'w': torch.tensor([1.])}, 'optimizer': {'step': torch.tensor(4)},
            'scheduler': {'last_epoch': 4}, 'split': {'train': [1]},
            'normalization': {'scale': [1.]}, 'torch_rng': torch.tensor([3], dtype=torch.uint8)}


def test_fm_branch_preserves_training_state_and_forces_diagonal():
    initial = payload()
    branched = copy.deepcopy(initial)
    assert module.branch_payload(branched) == {'w_lsd': 2., 'p_fm': .3, 'variant': 'rolling_lsd', 'velocity_mode': 'average'}
    assert branched['arguments']['w_lsd'] == branched['head_config']['w_lsd'] == 0.
    assert branched['arguments']['p_fm'] == branched['head_config']['p_fm'] == 1.
    assert branched['arguments']['variant'] == 'rolling_fm'
    assert branched['head_config']['velocity_mode'] == 'instant'
    for key in ('model', 'optimizer'):
        for name in initial[key]:
            assert torch.equal(initial[key][name], branched[key][name])
    assert torch.equal(initial['torch_rng'], branched['torch_rng'])
    for key in ('scheduler', 'split', 'normalization'):
        assert branched[key] == initial[key]


def test_fm_branch_rejects_inconsistent_or_incomplete_input():
    sample = payload()
    sample['head_config']['p_fm'] = .4
    with pytest.raises(ValueError, match='p_fm'):
        module.branch_payload(sample)
    sample = payload()
    sample['step'] = 5
    with pytest.raises(ValueError, match='boundary'):
        module.branch_payload(sample)
    sample = payload()
    del sample['torch_rng']
    with pytest.raises(ValueError, match='torch_rng'):
        module.branch_payload(sample)


def test_fm_branch_can_match_lsd_sampling_schedule():
    sample = payload()
    module.branch_payload(sample, match_time_sampling=True)
    assert sample['arguments']['p_fm'] == sample['head_config']['p_fm'] == .3
    assert sample['arguments']['w_lsd'] == sample['head_config']['w_lsd'] == 0.
    assert sample['arguments']['variant'] == 'rolling_fm'
