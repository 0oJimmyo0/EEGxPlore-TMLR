"""Compare prepared dataset split files/keys with those used in the paper."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='Prepared dataset directory')
    parser.add_argument('--dataset', type=str.upper, default='TUEV',
                        choices=['TUEV', 'FACED', 'ISRUC', 'PHYSIONET-MI', 'SEED-V'])
    args = parser.parse_args()
    manifest_dir = Path(__file__).resolve().parent / 'splits' / args.dataset
    manifest = json.loads((manifest_dir / 'manifest.json').read_text())
    errors = []
    lmdb_keys = None
    if args.dataset in {'FACED', 'PHYSIONET-MI', 'SEED-V'}:
        import lmdb
        import pickle
        env = lmdb.open(str(args.data), readonly=True, lock=False, readahead=False)
        with env.begin() as transaction:
            lmdb_keys = pickle.loads(transaction.get(b'__keys__'))
        env.close()
    for split, info in manifest.items():
        contents = (manifest_dir / (split + '.txt')).read_bytes()
        if hashlib.sha256(contents).hexdigest() != info['filename_list_sha256']:
            raise SystemExit('Changed reference filename list: ' + split)
        expected_order = contents.decode().splitlines()
        expected = set(expected_order)
        if lmdb_keys is not None:
            actual_order = [k.decode() if isinstance(k, bytes) else str(k) for k in lmdb_keys[split]]
            actual = set(actual_order)
            if actual_order != expected_order:
                errors.append(split + ' (key order)')
        elif args.dataset == 'ISRUC':
            actual = set()
            for subject in info['subjects']:
                subject_dir = args.data / 'seq' / f'ISRUC-group1-{subject}'
                for path in subject_dir.glob('*.npy'):
                    name = str(path.relative_to(args.data / 'seq'))
                    actual.add(name)
                    if not (args.data / 'labels' / name).is_file():
                        errors.append(split + ' (missing label file)')
        else:
            actual = {p.name for p in (args.data / info['directory']).glob('*.pkl')}
        missing, extra = expected - actual, actual - expected
        print(f'{split}: {len(actual)} files; missing={len(missing)}, unexpected={len(extra)}')
        if missing or extra:
            errors.append(split)
            print('  Missing examples:', sorted(missing)[:5])
            print('  Unexpected examples:', sorted(extra)[:5])
    if errors:
        raise SystemExit('Split filenames differ: ' + ', '.join(errors))
    print('Split files/keys match. Use the data audits to check signal shapes and labels.')


if __name__ == '__main__':
    main()
