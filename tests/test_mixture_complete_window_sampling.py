"""The mixture sampler must respect an explicitly filtered window index."""
import numpy as np
import pytest
import pickle
from accelerate import Accelerator
from torch.utils.data import DataLoader

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


class DatasetStub:
    def __init__(self, ids, lengths, steps, *, complete, horizon=4, name='stub'):
        self.trajectory_ids = np.asarray(ids)
        self.trajectory_lengths = np.asarray(lengths)
        self.all_steps = steps
        self.complete_windows_only = complete
        self.action_horizon = horizon
        self.dataset_name = name

    def __len__(self):
        return len(self.all_steps)


class ObservableMixture(LeRobotMixtureDataset):
    def __getitem__(self,index):
        _,episode,start=self.sample_step(index)
        return {'index':int(index),'epoch':self.epoch,'episode':int(episode),'start':int(start)}


def build(monkeypatch, datasets, weights, *, balance_trajectories=True, balance_datasets=False):
    monkeypatch.setattr(LeRobotMixtureDataset, 'update_metadata', lambda self, config: None)
    return LeRobotMixtureDataset(list(zip(datasets, weights)), mode='val',
        balance_dataset_weights=balance_datasets, balance_trajectory_weights=balance_trajectories)


@pytest.mark.parametrize('balance,expected', [(True, [.6,.4,0]), (False, [.5,.5,0])])
def test_complete_windows_obey_filtered_index_and_trajectory_weights(monkeypatch,balance,expected):
    allowed=[(10,0),(10,2),(10,6),(20,1),(20,3)]
    data=DatasetStub([10,20,30],[10,7,3],allowed,complete=True)
    mixture=build(monkeypatch,[data],[1.],balance_trajectories=balance)
    np.testing.assert_allclose(mixture.trajectory_sampling_weights[0],expected)
    observations=[mixture.sample_step(i) for i in range(2000)]
    assert {(int(episode),int(start)) for _,episode,start in observations} == set(allowed)
    assert all(start+4 <= data.trajectory_lengths[np.flatnonzero(data.trajectory_ids==episode)[0]]
               for _,episode,start in observations)
    np.testing.assert_array_equal(data.trajectory_lengths,[10,7,3])


def test_mixed_datasets_keep_ordinary_sampling_and_respect_filtered_index(monkeypatch):
    allowed=[(10,0),(10,2),(10,6)]
    restricted=DatasetStub([10],[10],allowed,complete=True,name='restricted')
    ordinary=DatasetStub([20],[5],[(20,i) for i in range(5)],complete=False,name='ordinary')
    mixture=build(monkeypatch,[restricted,ordinary],[.3,.7])
    observations=[mixture.sample_step(i) for i in range(2000)]
    assert {(int(e),int(s)) for d,e,s in observations if d is restricted} == set(allowed)
    assert {(int(e),int(s)) for d,e,s in observations if d is ordinary} == set(ordinary.all_steps)


def test_dataset_length_balancing_uses_filtered_window_count(monkeypatch):
    restricted=DatasetStub([10],[10],[(10,0),(10,2),(10,6)],complete=True)
    ordinary=DatasetStub([20],[5],[(20,i) for i in range(5)],complete=False)
    mixture=build(monkeypatch,[restricted,ordinary],[1.,1.],balance_datasets=True)
    np.testing.assert_allclose(mixture.dataset_sampling_weights,[3/8,5/8])


def test_ordinary_dataset_sampling_sequence_matches_legacy(monkeypatch):
    data=DatasetStub([10,20],[3,7],[(10,0),(20,0)],complete=False)
    mixture=build(monkeypatch,[data],[1.])
    for index in range(100):
        rng=np.random.default_rng(index)
        rng.choice(1,p=mixture.dataset_sampling_weights)
        trajectory=rng.choice(2,p=np.array([.3,.7]))
        expected=int(rng.choice(data.trajectory_lengths[trajectory]))
        _,episode,start=mixture.sample_step(index)
        assert int(episode)==int(data.trajectory_ids[trajectory])
        assert int(start)==expected


def test_empty_filtered_dataset_is_skipped(monkeypatch):
    empty=DatasetStub([10],[3],[],complete=True)
    ordinary=DatasetStub([20],[5],[(20,i) for i in range(5)],complete=False)
    mixture=build(monkeypatch,[empty,ordinary],[1.,1.])
    assert mixture.datasets==[ordinary]
    with pytest.raises(ValueError,match='No valid datasets'):
        build(monkeypatch,[empty],[1.])


@pytest.mark.parametrize('steps', [[(10,7)],[(30,0)]])
def test_invalid_filtered_index_fails_loudly(monkeypatch,steps):
    data=DatasetStub([10],[10],steps,complete=True)
    with pytest.raises(ValueError,match='Incomplete action window|Unknown trajectory'):
        build(monkeypatch,[data],[1.])


def test_old_serialized_mixture_requires_reinitialization_for_restricted_data(monkeypatch):
    data=DatasetStub([10],[10],[(10,0)],complete=True)
    mixture=build(monkeypatch,[data],[1.])
    del mixture._valid_start_indices
    with pytest.raises(RuntimeError,match='Recreate the mixture dataset'):
        mixture.sample_step(0)


def test_epoch_legacy_state_and_pickle(monkeypatch):
    data=DatasetStub([10],[10],[(10,i) for i in range(10)],complete=False)
    mixture=build(monkeypatch,[data],[1.])
    del mixture._shared_epoch
    mixture.__dict__['epoch']=7
    assert mixture.epoch==7
    mixture.set_epoch(8)
    assert mixture.epoch==8 and mixture._shared_epoch.is_shared()
    restored=pickle.loads(pickle.dumps(mixture))
    assert restored.epoch==8
    assert restored.sample_step(0)[1:]==mixture.sample_step(0)[1:]


def test_validation_sampling_remains_epoch_independent(monkeypatch):
    data=DatasetStub([10],[10],[(10,i) for i in range(10)],complete=False)
    mixture=build(monkeypatch,[data],[1.])
    before=[mixture.sample_step(i)[1:] for i in range(20)]
    mixture.set_epoch(9)
    assert [mixture.sample_step(i)[1:] for i in range(20)]==before


def test_accelerate_persistent_workers_observe_epoch_updates(monkeypatch):
    monkeypatch.setattr(LeRobotMixtureDataset,'update_metadata',lambda self,config:None)
    steps=[(10,i) for i in range(17)]+[(20,i) for i in range(23)]
    data=DatasetStub([10,20],[17,23],steps,complete=False)
    mixture=ObservableMixture([(data,1.)],mode='train',balance_dataset_weights=False,
                              balance_trajectory_weights=True,seed=42)
    def expected(epoch):
        mixture.set_epoch(epoch)
        return [(int(mixture.sample_step(i)[1]),int(mixture.sample_step(i)[2])) for i in range(len(mixture))]
    first_expected=expected(0);second_expected=expected(1);mixture.set_epoch(0)
    assert first_expected!=second_expected
    loader=Accelerator(cpu=True).prepare(DataLoader(mixture,batch_size=4,num_workers=2,
                                                   persistent_workers=True,shuffle=False))
    try:
        for epoch,reference in enumerate((first_expected,second_expected)):
            actual=[];seen_epochs=[]
            for batch in loader:
                actual.extend(zip(batch['episode'].tolist(),batch['start'].tolist()))
                seen_epochs.extend(batch['epoch'].tolist())
            assert seen_epochs==[epoch]*len(reference)
            assert actual==reference
    finally:
        base=getattr(loader,'base_dataloader',loader)
        iterator=getattr(base,'_iterator',None)
        if iterator is not None:iterator._shutdown_workers()
