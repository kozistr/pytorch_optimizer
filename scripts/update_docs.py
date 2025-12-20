"""Automatically update documentation files with new optimizers, schedulers, and losses."""

import re
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


def parse_init_imports(init_content: str) -> dict[str, list[str]]:
    """Parse __init__.py to extract imports by module."""
    imports: dict[str, list[str]] = {'loss': [], 'lr_scheduler': [], 'optimizer': [], 'optimizer.utils': []}
    pattern = re.compile(r'from pytorch_optimizer\.(\w+(?:\.\w+)?)\s+import\s+\(([^)]+)\)', re.DOTALL)

    for match in pattern.finditer(init_content):
        module, items_str = match.group(1), match.group(2)
        if module in imports:
            for item in items_str.split(','):
                name = item.split('#')[0].strip()
                if name:
                    imports[module].append(name)

    return imports


def generate_docs(title: str, header: list[str], items: list[str], excludes: set[str]) -> str:
    """Generate markdown documentation content."""
    header_names = {h.split('.')[-1] for h in header}
    body_items = sorted(i.lower() for i in items if i not in excludes and i not in header_names)

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
    init_content = (root_dir / 'pytorch_optimizer' / '__init__.py').read_text(encoding='utf-8')
    imports = parse_init_imports(init_content)
    docs_dir = root_dir / 'docs'

    configs = [
        ('optimizer.md', 'Optimizers', OPTIMIZER_HEADER, imports['optimizer'], OPTIMIZER_EXCLUDES),
        (
            'lr_scheduler.md',
            'Learning Rate Scheduler',
            LR_SCHEDULER_HEADER,
            imports['lr_scheduler'],
            LR_SCHEDULER_EXCLUDES,
        ),
        ('loss.md', 'Loss Function', LOSS_HEADER, imports['loss'], LOSS_EXCLUDES),
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
