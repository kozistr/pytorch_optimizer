from scripts.update_docs import update_docs


def test_update_docs_from_public_exports(tmp_path):
    exports = {
        'optimizer': ['Adam', 'Lion', 'create_optimizer'],
        'lr_scheduler': ['StepLR', 'CosineScheduler', 'get_wsd_schedule'],
        'loss': ['SoftF1Loss', 'BCELoss', 'bi_tempered_logistic_loss'],
    }

    for module, names in exports.items():
        module_dir = tmp_path / 'pytorch_optimizer' / module
        module_dir.mkdir(parents=True)
        (module_dir / '__init__.py').write_text(
            f'__all__ = {names!r}\nraise RuntimeError(\'Package dependencies must not be imported\')\n',
            encoding='utf-8',
        )

    docs_dir = tmp_path / 'docs'
    docs_dir.mkdir()

    assert update_docs(tmp_path) == ['docs/optimizer.md', 'docs/lr_scheduler.md', 'docs/loss.md']

    optimizer_docs = (docs_dir / 'optimizer.md').read_text(encoding='utf-8')
    assert '::: pytorch_optimizer.Lion\n' in optimizer_docs
    assert '::: pytorch_optimizer.Adam\n' not in optimizer_docs
    assert optimizer_docs.count('::: pytorch_optimizer.optimizer.create_optimizer\n') == 1

    scheduler_docs = (docs_dir / 'lr_scheduler.md').read_text(encoding='utf-8')
    assert '::: pytorch_optimizer.CosineScheduler\n' in scheduler_docs
    assert '::: pytorch_optimizer.StepLR\n' not in scheduler_docs
    assert scheduler_docs.count('::: pytorch_optimizer.get_wsd_schedule\n') == 1

    loss_docs = (docs_dir / 'loss.md').read_text(encoding='utf-8')
    assert loss_docs.count('::: pytorch_optimizer.bi_tempered_logistic_loss\n') == 1
    assert loss_docs.index('::: pytorch_optimizer.BCELoss\n') < loss_docs.index('::: pytorch_optimizer.SoftF1Loss\n')

    assert update_docs(tmp_path) == []
