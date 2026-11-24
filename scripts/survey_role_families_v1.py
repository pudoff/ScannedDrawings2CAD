"""Rebuild the train/validation native-text screen for P14.1b.10b."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--catalog', default='artifacts/dataset-content-review-v1/catalog.json')
    parser.add_argument('--catalog-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.setdefault('XDG_CACHE_HOME', str(args.workspace/'artifacts/.cache'))
    from cad_recovery.dataset_layout_v1 import checked, encoded, inside, pin, read
    from cad_recovery.family_catalog_v2 import require
    from cad_recovery.role_admission_native_v1 import survey_native
    root = args.workspace.resolve()
    output = inside(root, args.output)
    require(not output.exists() and not output.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a new output outside dataset')
    catalog_pin = {'path': args.catalog, 'sha256': args.catalog_sha256}
    rows = survey_native(root, read(checked(root, catalog_pin)), lambda n, total: print(f'{n}/{total}', flush=True))
    output.mkdir(parents=True, exist_ok=False)
    (output/'survey.json').write_bytes(encoded(rows))
    code = ['src/cad_recovery/role_admission_native_v1.py', 'scripts/survey_role_families_v1.py', 'requirements.lock']
    (output/'manifest.json').write_bytes(encoded({'schema_version': 'role-family-survey-manifest/1.0',
        'catalog': catalog_pin, 'code': [pin(root, name) for name in code], 'test_content_read': False,
        'independent_evaluation': False, 'files': [pin(output, 'survey.json')]}))
    print(pin(root, (output/'manifest.json').relative_to(root).as_posix()))


if __name__ == '__main__':
    main()
