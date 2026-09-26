import argparse
import re
from pathlib import Path


def update_changelog(root_dir: Path, version: str, notes: str) -> None:
    if not re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', version):
        raise ValueError('Release tag must have the form vMAJOR.MINOR.PATCH')
    if not notes.strip():
        raise ValueError('Release notes must not be empty')

    changelog = root_dir / 'CHANGELOG.md'
    original = changelog.read_text(encoding='utf-8')
    entry = f'# {version}\n\n{notes.strip()}\n\n'
    section = re.compile(
        rf'^# {re.escape(version)}[ \t]*\n.*?(?=^# v[0-9]+\.[0-9]+\.[0-9]+[ \t]*$|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    updated, count = section.subn(lambda _: entry, original, count=1)

    docs_dir = root_dir / 'docs' / 'changelogs'
    (docs_dir / f'{version}.md').write_text(entry.rstrip() + '\n', encoding='utf-8')
    changelog.write_text(updated if count else entry + original, encoding='utf-8')

    versions = sorted(
        docs_dir.glob('v*.md'), key=lambda path: tuple(map(int, path.stem[1:].split('.'))), reverse=True
    )
    index = '# Changelog\n\nRelease notes, newest first.\n\n'
    index += ''.join(f'- [{path.stem}]({path.name})\n' for path in versions)
    (docs_dir / 'index.md').write_text(index, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('version')
    parser.add_argument('notes', type=Path)
    args = parser.parse_args()

    update_changelog(Path.cwd(), args.version, args.notes.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
