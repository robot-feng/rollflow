import copy
import importlib.util
from pathlib import Path

import pytest
import torch

path = (Path(__file__).resolve().parents[1] /
        'examples/simBenchmarks/Minari/train_files/branch_lsd.py')
spec = importlib.util.spec_from_file_location('minari_lsd_branch', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_payload():
    return {'arguments': {'variant': 'rolling_lsd', 'fm_warmup': 4, 'w_lsd': .05, 'p_k1': 0.},
            'head_config': {'w_lsd': .05, 'p_k1': 0.}, 'step': 4,
            'model': {'weight': torch.tensor([1.])},
            'optimizer': {'step': torch.tensor(4)}, 'scheduler': {'last_epoch': 4},
            'split': {'train': [1]}, 'normalization': {'scale': [1.]}}


def test_branch_only_changes_weight_configuration():
    source = sample_payload()
    payload = copy.deepcopy(source)
    assert module.branch_payload(payload, .2) == .05
    assert payload['arguments']['w_lsd'] == payload['head_config']['w_lsd'] == .2
    assert torch.equal(payload['model']['weight'], source['model']['weight'])
    assert torch.equal(payload['optimizer']['step'], source['optimizer']['step'])
    assert payload['scheduler'] == source['scheduler']
    assert payload['split'] == source['split']
    assert payload['normalization'] == source['normalization']


def test_branch_rejects_wrong_step_or_bad_weight():
    payload = sample_payload()
    payload['step'] = 5
    with pytest.raises(ValueError, match='boundary'):
        module.branch_payload(payload, .2)
    with pytest.raises(ValueError, match='finite and positive'):
        module.branch_payload(sample_payload(), float('nan'))


def test_branch_can_change_group_sampling_explicitly():
    payload = sample_payload()
    module.branch_payload(payload, .2, p_k1=.5)
    assert payload['arguments']['p_k1'] == payload['head_config']['p_k1'] == .5
    with pytest.raises(ValueError, match='p_k1'):
        module.branch_payload(sample_payload(), .2, p_k1=1.1)


def test_branch_can_record_precision_without_changing_model():
    payload=sample_payload()
    original=copy.deepcopy(payload)
    module.branch_payload(payload,.2,precision='bf16')
    assert payload['arguments']['precision']=='bf16'
    assert torch.equal(payload['model']['weight'],original['model']['weight'])
    assert torch.equal(payload['optimizer']['step'],original['optimizer']['step'])
    with pytest.raises(ValueError,match='precision'):
        module.branch_payload(sample_payload(),.2,precision='fp16')
