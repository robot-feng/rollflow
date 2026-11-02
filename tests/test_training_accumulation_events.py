"""Exercise the real trainer loop at a model-only resume boundary on CPU."""
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from torch import nn

from starVLA.training.train_starvla import VLATrainer


@pytest.mark.parametrize('initial, expected_eval', [(0, 2), (2, 4)])
def test_step_events_only_on_accumulation_boundaries(initial, expected_eval):
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.completed_steps = initial
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(
        max_train_steps=initial+3, eval_interval=2, save_interval=2))
    trainer.accelerator = SimpleNamespace(sync_gradients=False, is_local_main_process=False)
    events = []
    micro_steps = 0
    batches = 0

    trainer._log_training_config = lambda: None
    trainer._create_data_iterators = lambda: None

    def next_batch():
        nonlocal batches
        value = batches
        batches += 1
        return value

    def train_step(batch):
        nonlocal micro_steps
        micro_steps += 1
        trainer.accelerator.sync_gradients = micro_steps % 2 == 0
        events.append(('micro', trainer.completed_steps, batch, trainer.accelerator.sync_gradients))
        return {'loss': 0.0}

    def eval_step(metrics):
        events.append(('eval', trainer.completed_steps, next_batch()))
        return metrics

    trainer._get_next_batch = next_batch
    trainer._train_step = train_step
    trainer.eval_action_model = eval_step
    trainer._log_metrics = lambda metrics: events.append(('log', trainer.completed_steps))
    trainer._save_checkpoint = lambda: events.append(('save', trainer.completed_steps))
    trainer._finalize_training = lambda: events.append(('finalize', trainer.completed_steps))
    trainer.train()

    assert trainer.completed_steps == initial + 3
    assert micro_steps == 6
    assert batches == 7  # Six train microbatches and one evaluation batch.
    assert [e[1] for e in events if e[0] == 'log'] == list(range(initial+1, initial+4))
    assert [e[1] for e in events if e[0] == 'eval'] == [expected_eval]
    assert [e[1] for e in events if e[0] == 'save'] == [expected_eval]
    assert events[0] == ('micro', initial, 0, False)
    assert events[-1] == ('finalize', initial+3)


def test_real_cpu_accelerator_accumulates_and_gates_events():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, examples, training_step):
            del training_step
            return {'action_loss': self.linear(examples).square().mean()}

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.completed_steps = 2
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(
        max_train_steps=5, eval_interval=2, save_interval=2, gradient_clipping=None))
    trainer.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.model, trainer.optimizer = trainer.accelerator.prepare(model, optimizer)
    events = []
    trainer.lr_scheduler = SimpleNamespace(step=lambda: events.append(('scheduler', trainer.completed_steps)))
    trainer._log_training_config = lambda: None
    trainer._create_data_iterators = lambda: None
    trainer._get_next_batch = lambda: torch.ones(1, 1)
    train_step = trainer._train_step

    def observed_train_step(batch):
        result = train_step(batch)
        events.append(('micro', trainer.completed_steps, trainer.accelerator.sync_gradients))
        return result

    trainer._train_step = observed_train_step
    trainer.eval_action_model = lambda metrics: events.append(('eval', trainer.completed_steps)) or metrics
    trainer._log_metrics = lambda metrics: events.append(('log', trainer.completed_steps))
    trainer._save_checkpoint = lambda: events.append(('save', trainer.completed_steps))
    trainer._finalize_training = lambda: None
    trainer.train()

    assert [x[2] for x in events if x[0] == 'micro'] == [False, True] * 3
    assert [x[1] for x in events if x[0] == 'eval'] == [4]
    assert [x[1] for x in events if x[0] == 'save'] == [4]
    assert [x[1] for x in events if x[0] == 'log'] == [3, 4, 5]
    assert len([x for x in events if x[0] == 'scheduler']) == 3
    assert all(int(state['step']) == 3 for state in trainer.optimizer.state.values())


def test_real_cpu_accelerator_keeps_microbatch_gradients_until_sync():
    class SignedLossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, examples, training_step):
            del training_step
            return {'action_loss': self.weight * examples}

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=0.5))
    trainer.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model = SignedLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.model, trainer.optimizer = trainer.accelerator.prepare(model, optimizer)
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)

    # Each backward contributes +1 / -1 after Accelerate's 1/2 scaling.
    # The correct accumulated update is zero. Clipping either partial gradient
    # or zeroing immediately before the second backward changes that result.
    trainer._train_step(torch.tensor(2.0))
    trainer._train_step(torch.tensor(-2.0))

    assert torch.equal(trainer.model.weight.detach(), torch.tensor(0.0))
