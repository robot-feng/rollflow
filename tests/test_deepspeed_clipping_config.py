from types import SimpleNamespace

import pytest
from accelerate import DeepSpeedPlugin
from accelerate.utils import DistributedType

from starVLA.training.train_starvla import VLATrainer, _configure_deepspeed_gradient_clipping


def _accelerator(plugin):
    return SimpleNamespace(
        distributed_type=DistributedType.DEEPSPEED,
        state=SimpleNamespace(deepspeed_plugin=plugin),
    )


def _plugin(monkeypatch):
    monkeypatch.delenv("ACCELERATE_GRADIENT_CLIPPING", raising=False)
    return DeepSpeedPlugin(gradient_accumulation_steps=1, zero_stage=2)


def test_accelerate_default_auto_ignores_trainer_setting(monkeypatch):
    plugin = _plugin(monkeypatch)
    assert plugin.get_value("gradient_clipping") == "auto"
    # Accelerate 1.13 fills this argument with 1.0 during prepare().
    plugin.fill_match("gradient_clipping", must_match=False, gradient_clipping=1.0)
    assert plugin.get_value("gradient_clipping") == 1.0


@pytest.mark.parametrize("value,expected", [(1.0, 1.0), (0.25, 0.25), (None, 0.0)])
def test_trainer_value_survives_accelerate_auto_processing(monkeypatch, value, expected):
    plugin = _plugin(monkeypatch)
    _configure_deepspeed_gradient_clipping(_accelerator(plugin), value)
    # This operates on the same configuration that the plugin passes to DeepSpeed.
    plugin.fill_match("gradient_clipping", must_match=False, gradient_clipping=1.0)
    assert plugin.get_value("gradient_clipping") == expected
    assert plugin.hf_ds_config.config["gradient_clipping"] == expected


def test_missing_key_is_filled_and_matching_explicit_value_is_preserved(monkeypatch):
    plugin = _plugin(monkeypatch)
    plugin.deepspeed_config.pop("gradient_clipping")
    _configure_deepspeed_gradient_clipping(_accelerator(plugin), 0.25)
    assert plugin.get_value("gradient_clipping") == 0.25
    _configure_deepspeed_gradient_clipping(_accelerator(plugin), 0.25)
    assert plugin.get_value("gradient_clipping") == 0.25


def test_conflicting_explicit_deepspeed_value_fails(monkeypatch):
    plugin = _plugin(monkeypatch)
    plugin.deepspeed_config["gradient_clipping"] = 0.5
    with pytest.raises(ValueError, match="conflicts"):
        _configure_deepspeed_gradient_clipping(_accelerator(plugin), 0.25)
    assert plugin.get_value("gradient_clipping") == 0.5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_trainer_value_fails(monkeypatch, value):
    with pytest.raises(ValueError, match="positive and finite"):
        _configure_deepspeed_gradient_clipping(_accelerator(_plugin(monkeypatch)), value)


def test_other_backends_are_untouched():
    accelerator = SimpleNamespace(distributed_type=DistributedType.NO)
    _configure_deepspeed_gradient_clipping(accelerator, 0.25)


def test_trainer_applies_threshold_before_distributed_prepare(monkeypatch):
    plugin = _plugin(monkeypatch)
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(seed=42, trainer=SimpleNamespace(
        freeze_modules=None, gradient_clipping=0.25,
    ))
    trainer.accelerator = _accelerator(plugin)
    trainer.model, trainer.optimizer, trainer.vla_train_dataloader = object(), object(), object()
    trainer._save_initial_configs = lambda: None
    trainer._init_checkpointing = lambda: None
    trainer._adjust_lr_scheduler_for_resume = lambda: None
    trainer.freeze_backbones = lambda model, **kwargs: model
    trainer.print_trainable_parameters = lambda model: None
    trainer._init_wandb = lambda: None

    def prepare(accelerator, *components):
        assert plugin.get_value("gradient_clipping") == 0.25
        # Accelerate's auto processing must preserve the configured threshold.
        plugin.fill_match("gradient_clipping", must_match=False, gradient_clipping=1.0)
        assert plugin.get_value("gradient_clipping") == 0.25
        return components

    trainer.setup_distributed_training = prepare
    trainer.prepare_training()
