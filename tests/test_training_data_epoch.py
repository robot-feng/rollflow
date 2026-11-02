from types import SimpleNamespace

import numpy as np
from torch import nn

from starVLA.training.train_starvla import VLATrainer


class _ThreeBatchLoader:
    def __len__(self):
        return 3

    def __iter__(self):
        for index in range(3):
            yield [{'id':index,'action':np.zeros((4,1),dtype=np.float32)}]


class _Policy(nn.Module):
    action_horizon=4

    def predict_action(self,examples,**kwargs):
        return {'normalized_actions':np.zeros((len(examples),1,1),dtype=np.float32)}


def test_data_epoch_counts_train_eval_and_restart_without_changing_legacy_metric():
    trainer=VLATrainer.__new__(VLATrainer)
    trainer.vla_train_dataloader=_ThreeBatchLoader()
    trainer.config=SimpleNamespace(trainer=SimpleNamespace(logging_frequency=1))
    trainer.completed_steps=5000  # A model-only resume need not restore the data position.
    trainer.optimizer=SimpleNamespace(param_groups=[{'name':'head'}])
    trainer.lr_scheduler=SimpleNamespace(get_last_lr=lambda:[1e-4])
    trainer.accelerator=SimpleNamespace(is_main_process=True,unwrap_model=lambda model:model)
    trainer.model=_Policy()
    trainer._wandb_enabled=False
    trainer._create_data_iterators()

    assert trainer._get_next_batch()[0]['id']==0
    trainer.eval_action_model({})  # Consumes the second batch through the same iterator.
    metrics={};trainer._log_metrics(metrics)
    assert metrics['data_epoch_since_launch']==0.6667
    assert metrics['epoch']==round(5000/3,2)

    assert trainer._get_next_batch()[0]['id']==2
    metrics={};trainer._log_metrics(metrics)
    assert metrics['data_epoch_since_launch']==1.0
    assert trainer._get_next_batch()[0]['id']==0
    metrics={};trainer._log_metrics(metrics)
    assert metrics['data_epoch_since_launch']==1.3333

    trainer._create_data_iterators()
    metrics={};trainer._log_metrics(metrics)
    assert metrics['data_epoch_since_launch']==0.0
