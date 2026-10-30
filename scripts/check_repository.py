"""Check portable links, notebook structure and the publication file boundary."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv', 'build', 'dist'}
SECRET_PATTERNS = [r'\bgh[pousr]_[A-Za-z0-9]{25,}', r'\bgithub_pat_[A-Za-z0-9_]{30,}',
                   r'\bhf_[A-Za-z0-9]{30,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']


def check():
    failures = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and not IGNORE.intersection(p.relative_to(ROOT).parts)
             and '.egg-info' not in str(p)]
    notebooks = 0
    for p in files:
        relative = p.relative_to(ROOT)
        if p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.nii', '.npz', '.npy', '.lnk'} or p.name.endswith('.nii.gz'):
            failures.append(f'Unwanted large/local artifact: {relative}')
        if p.stat().st_size > 20 * 1024 * 1024:
            failures.append(f'Unexpected file above 20 MiB: {relative}')
        if p.name in {'.env', 'kaggle.json'} or p.suffix in {'.pem', '.key'}:
            failures.append(f'Credential file: {relative}')
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
        if p.suffix == '.md':
            content = re.sub(r'```[\s\S]*?```', '', p.read_text(encoding='utf-8'))
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
                target = target.strip('<> ').split('#')[0]
                if not target or re.match(r'^[a-z]+:', target):
                    continue
                if not (p.parent / unquote(target)).exists():
                    failures.append(f'Broken local link in {relative}: {target}')
        if p.suffix == '.ipynb':
            notebooks += 1
            n = json.loads(p.read_text(encoding='utf-8'))
            if n.get('nbformat') != 4:
                failures.append(f'Invalid notebook version: {relative}')
            for c in n['cells']:
                if c['cell_type'] == 'code':
                    ast.parse(''.join(c['source']))
                    if p.name == 'BTP1_GPU_Run_Notebook.ipynb' and (c['outputs'] or c['execution_count'] is not None):
                        failures.append(f'Fresh-run notebook has execution output: {relative}')
            scan = '\n'.join(''.join(c['source']) for c in n['cells'])
            scan += '\n'.join(''.join(o.get('text', [])) for c in n['cells'] for o in c.get('outputs', []))
        elif p.suffix in {'.py', '.md', '.json', '.yml', '.yaml', '.txt', '.toml', '.cff', '.tex'}:
            scan = p.read_text(encoding='utf-8-sig')
        else:
            scan = ''
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, scan):
                failures.append(f'Possible credential pattern in {relative}')
        if p.suffix in {'.py', '.md', '.yml', '.toml', '.tex'} and re.search(r'[A-Z]:[\\/]Users[\\/]', scan):
            failures.append(f'Machine-specific Windows home path in {relative}')
    bundle = ROOT / 'notebooks/btp1_gpu_code.zip'
    with ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            failures.append('Corrupt Colab bundle')
        for p in (ROOT / 'src/btp3d').glob('*.py'):
            rel = p.relative_to(ROOT).as_posix()
            if archive.read(rel).replace(b'\r\n', b'\n') != p.read_bytes().replace(b'\r\n', b'\n'):
                failures.append(f'Stale bundled module: {rel}')
        if archive.read('README.md').replace(b'\r\n', b'\n') != (ROOT / 'README.md').read_bytes().replace(b'\r\n', b'\n'):
            failures.append('Stale bundled README')
    if notebooks != 3:
        failures.append(f'Expected 3 notebooks, found {notebooks}')
    if failures:
        raise ValueError('\n'.join(failures))
    return {'passed': True, 'checked_files': len(files), 'notebooks': notebooks,
            'local_links_valid': True, 'forbidden_artifacts': 0, 'credential_patterns': 0,
            'colab_bundle_current': True}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
