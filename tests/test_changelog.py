import runpy
import sys
from pathlib import Path

import pytest
from scripts.update_changelog import update_changelog


@pytest.fixture
def changelog_root(tmp_path):
    docs_dir = tmp_path / 'docs' / 'changelogs'
    docs_dir.mkdir(parents=True)
    for version in ('v3.10.1', 'v3.9.0'):
        (docs_dir / f'{version}.md').write_text(f'# {version}\n\nOriginal notes\n', encoding='utf-8')
    (tmp_path / 'CHANGELOG.md').write_text(
        '# v3.10.1\n\nOriginal notes\n\n# v3.9.0\n\nOriginal notes\n', encoding='utf-8'
    )
    return tmp_path


def test_update_changelog_preserves_history_and_is_repeatable(changelog_root):
    changelog = changelog_root / 'CHANGELOG.md'
    original = changelog.read_text(encoding='utf-8')
    docs_dir = changelog_root / 'docs' / 'changelogs'
    (docs_dir / 'v3.10.2.md').write_text('Draft notes', encoding='utf-8')
    notes = "## What's Changed\n\n* Fix training by @author in https://github.com/kozistr/pytorch_optimizer/pull/529\n"

    update_changelog(changelog_root, 'v3.10.2', notes)

    assert changelog.read_text(encoding='utf-8') == f'# v3.10.2\n\n{notes}\n{original}'
    assert (docs_dir / 'v3.10.2.md').read_text(encoding='utf-8') == f'# v3.10.2\n\n{notes}'
    assert (docs_dir / 'index.md').read_text(encoding='utf-8').splitlines()[4:] == [
        '- [v3.10.2](v3.10.2.md)',
        '- [v3.10.1](v3.10.1.md)',
        '- [v3.9.0](v3.9.0.md)',
    ]

    snapshot = {path: path.read_bytes() for path in changelog_root.rglob('*.md')}
    update_changelog(changelog_root, 'v3.10.2', notes)
    assert snapshot == {path: path.read_bytes() for path in changelog_root.rglob('*.md')}


@pytest.mark.parametrize('version', ['v3.10.1', 'v3.9.0'])
def test_update_changelog_replaces_only_the_matching_release(changelog_root, version):
    notes = '## Fixes\n\n* Preserve C:\\models\\1 and Unicode: β.\n'
    update_changelog(changelog_root, version, notes)

    changelog = (changelog_root / 'CHANGELOG.md').read_text(encoding='utf-8')
    other_version = 'v3.9.0' if version == 'v3.10.1' else 'v3.10.1'
    assert changelog.count(f'# {version}\n') == 1
    assert f'# {version}\n\n{notes}' in changelog
    assert f'# {other_version}\n\nOriginal notes\n' in changelog
    assert changelog.index('# v3.10.1') < changelog.index('# v3.9.0')
    assert (changelog_root / 'docs' / 'changelogs' / f'{version}.md').read_text(encoding='utf-8') == (
        f'# {version}\n\n{notes}'
    )


@pytest.mark.parametrize(
    ('version', 'notes'),
    [('3.10.2', 'Notes'), ('../../outside', 'Notes'), ('v3.10.2', ''), ('v3.10.2', ' \n')],
)
def test_update_changelog_rejects_invalid_input_without_writing(changelog_root, version, notes):
    snapshot = {path: path.read_bytes() for path in changelog_root.rglob('*.md')}
    with pytest.raises(ValueError):
        update_changelog(changelog_root, version, notes)
    assert snapshot == {path: path.read_bytes() for path in changelog_root.rglob('*.md')}


def test_update_changelog_cli(changelog_root, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'update_changelog.py'
    notes = changelog_root / 'release-notes.md'
    notes.write_text('## Changes\n\n* Generated notes\n', encoding='utf-8')
    monkeypatch.chdir(changelog_root)
    monkeypatch.setattr(sys, 'argv', [str(script), 'v3.10.2', str(notes)])

    runpy.run_path(str(script), run_name='__main__')

    assert (changelog_root / 'CHANGELOG.md').read_text(encoding='utf-8').startswith(
        '# v3.10.2\n\n## Changes\n\n* Generated notes\n\n# v3.10.1\n'
    )
