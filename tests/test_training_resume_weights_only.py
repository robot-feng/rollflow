from types import SimpleNamespace

import starVLA.training.train_starvla as training


def test_model_only_resume_reports_missing_training_state(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(training, 'logger', SimpleNamespace(info=lambda message: None, warning=messages.append))
    model = object()
    loaded = object()
    trainer = SimpleNamespace(
        config=SimpleNamespace(output_dir=str(tmp_path), trainer=SimpleNamespace(
            pretrained_checkpoint=None, is_resume=True)),
        model=model,
        completed_steps=0,
        _get_latest_checkpoint=lambda path: ('checkpoint.pt', 1000),
        load_pretrained_backbones=lambda original, path, reload_modules: loaded,
    )
    training.VLATrainer._init_checkpointing(trainer)
    assert trainer.model is loaded
    assert trainer.completed_steps == 1000
    assert len(messages) == 1
    assert 'optimizer state' in messages[0] and 'not restored' in messages[0]
