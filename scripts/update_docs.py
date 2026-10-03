"""Automatically update documentation files with new optimizers, schedulers, and losses."""

import ast
import sys
from pathlib import Path

OPTIMIZER_EXCLUDES = {'Adam', 'AdamW', 'SGD', 'NAdam', 'RMSprop', 'LBFGS'}
LR_SCHEDULER_EXCLUDES = {
    'ConstantLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'CyclicLR',
    'MultiplicativeLR',
    'MultiStepLR',
    'OneCycleLR',
    'StepLR',
    'get_chebyshev_perm_steps',
}
LOSS_EXCLUDES = set()

OPTIMIZER_HEADER = [
    'pytorch_optimizer.optimizer.create_optimizer',
    'pytorch_optimizer.optimizer.get_optimizer_parameters',
]
LR_SCHEDULER_HEADER = [
    'pytorch_optimizer.deberta_v3_large_lr_scheduler',
    'pytorch_optimizer.get_chebyshev_schedule',
    'pytorch_optimizer.get_wsd_schedule',
]
LOSS_HEADER = ['pytorch_optimizer.bi_tempered_logistic_loss']


def parse_init_exports(root_dir: Path) -> dict[str, list[str]]:
    """Read public exports without importing the package or its dependencies."""
    exports: dict[str, list[str]] = {}
    for module in ('loss', 'lr_scheduler', 'optimizer'):
        init_path = root_dir / 'pytorch_optimizer' / module / '__init__.py'
        tree = ast.parse(init_path.read_text(encoding='utf-8'))
        value = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets)
        )
        exports[module] = ast.literal_eval(value)

    return exports


def generate_docs(title: str, header: list[str], items: list[str], excludes: set[str]) -> str:
    """Generate markdown documentation content."""
    header_names = {h.split('.')[-1] for h in header}
    body_items = sorted((i for i in items if i not in excludes and i not in header_names), key=str.lower)

    lines = [f'# {title}', '']
    for item in header:
        lines.append(f'::: {item}\n    :docstring:\n    :members:\n')
    for item in body_items:
        lines.append(f'::: pytorch_optimizer.{item}\n    :docstring:\n    :members:\n')

    return '\n'.join(lines)


def write_if_changed(file_path: Path, content: str) -> bool:
    """Write content only if it differs from current content."""
    if file_path.exists() and file_path.read_text(encoding='utf-8').strip() == content.strip():
        return False
    file_path.write_text(content, encoding='utf-8')
    return True


def update_docs(root_dir: Path) -> list[str]:
    """Update all documentation files. Returns list of changed files."""
    exports = parse_init_exports(root_dir)
    docs_dir = root_dir / 'docs'

    configs = [
        ('optimizer.md', 'Optimizers', OPTIMIZER_HEADER, exports['optimizer'], OPTIMIZER_EXCLUDES),
        (
            'lr_scheduler.md',
            'Learning Rate Scheduler',
            LR_SCHEDULER_HEADER,
            exports['lr_scheduler'],
            LR_SCHEDULER_EXCLUDES,
        ),
        ('loss.md', 'Loss Function', LOSS_HEADER, exports['loss'], LOSS_EXCLUDES),
    ]

    return [
        f'docs/{filename}'
        for filename, title, header, items, excludes in configs
        if write_if_changed(docs_dir / filename, generate_docs(title, header, items, excludes))
    ]


def main():
    root_dir = Path(__file__).parent.parent

    if not (root_dir / 'pytorch_optimizer' / '__init__.py').exists():
        print('Error: Could not find pytorch_optimizer package', file=sys.stderr)
        return

    changed_files = update_docs(root_dir)
    if changed_files:
        print('Updated documentation files:')
        for f in changed_files:
            print(f'  - {f}')
    else:
        print('No documentation changes needed.')


if __name__ == '__main__':
    main()
