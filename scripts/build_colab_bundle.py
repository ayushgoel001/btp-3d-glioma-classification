"""Build the small source/label ZIP expected by the Colab fresh-run notebook."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def build_bundle():
    output = ROOT / 'notebooks/btp1_gpu_code.zip'
    paths = list((ROOT / 'src/btp3d').glob('*.py'))
    paths += list((ROOT / 'configs').glob('*.json'))
    paths += [ROOT / x for x in ['scripts/download_brats2020.py', 'pyproject.toml', 'requirements.txt', 'README.md']]
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.write(p, p.relative_to(ROOT).as_posix())
        archive.write(ROOT / 'data/metadata/name_mapping.csv', 'labels/name_mapping.csv')
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid code ZIP')
    return output


if __name__ == '__main__':
    print(build_bundle())
