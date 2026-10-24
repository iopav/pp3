"""Remove explicitly named CSV columns without converting retained cell values.

Preview: python scripts/clean_metrics_csv.py ROOT --drop COLUMN [COLUMN ...]
Apply:   append --apply; each changed file gets an exact .csv.bak backup.
PP3 metadata/diagnostics only: replace --drop ... with --pp3-metadata.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


# Exact names only: keep epoch and every measured numeric metric, including
# known/estimated FLOPs and data-wait times. Identity remains in directory paths.
PP3_METADATA = {
    'method', 'dataset', 'seed', 'unit', 'source_unit', 'smoke',
    'train_energy_method', 'train_energy_reason', 'train_complete',
    'test_energy_method', 'test_energy_reason', 'test_complete',
    'epoch_flops_complete',
}


def prepare(path, columns):
    original = path.read_bytes()
    bom = original.startswith(b'\xef\xbb\xbf')
    content = original.decode('utf-8-sig')
    rows = list(csv.reader(io.StringIO(content, newline=''), strict=True))
    if not rows or not rows[0] or len(set(rows[0])) != len(rows[0]):
        raise ValueError(f'Empty or duplicate header: {path}')
    header = rows[0]
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError(f'Inconsistent row width: {path}')
    removed = [name for name in header if name in columns]
    keep = [i for i, name in enumerate(header) if name not in columns]
    if not keep:
        raise ValueError(f'Refusing to remove every column: {path}')
    expected = [[row[i] for i in keep] for row in rows]
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\r\n' if '\r\n' in content else '\n').writerows(expected)
    rendered = stream.getvalue()
    assert list(csv.reader(io.StringIO(rendered, newline=''))) == expected
    encoded = rendered.encode('utf-8-sig' if bom else 'utf-8')
    return original, encoded, dict(path=str(path), rows=len(rows)-1, removed=removed,
        retained_columns=[header[i] for i in keep],
        original_sha256=hashlib.sha256(original).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--drop', nargs='+')
    selection.add_argument('--pp3-metadata', action='store_true',
                           help='Remove PP3 metadata/diagnostics; preserve all numeric metrics')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.drop = sorted(PP3_METADATA) if args.pp3_metadata else args.drop
    root = args.root.resolve(strict=True)
    if not root.is_dir():
        parser.error('root must be a directory')
    files = sorted(root.rglob('metrics.csv'))
    if not files:
        parser.error('No metrics.csv found')
    plans = []
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'File outside root or symlink: {path}')
        original, encoded, info = prepare(path, set(args.drop))
        backup = path.with_suffix('.csv.bak')
        if args.apply and info['removed'] and backup.exists():
            raise FileExistsError(f'Backup already exists; refusing to overwrite: {backup}')
        plans.append((path, backup, original, encoded, info))
    # Validate every input before touching any original file.
    for path, backup, original, encoded, info in plans:
        info['applied'] = False
        if args.apply and info['removed']:
            if path.read_bytes() != original:
                raise RuntimeError(f'File changed after preview: {path}')
            with backup.open('xb') as stream:
                stream.write(original)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(encoded)
                os.replace(temporary, path)
                assert path.read_bytes() == encoded
                assert backup.read_bytes() == original
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            info['applied'] = True
        print(json.dumps({k: info[k] for k in ('path', 'rows', 'removed', 'applied')}, ensure_ascii=False))
    report = dict(files=len(plans), changed=sum(p[4]['applied'] for p in plans),
                  matching=sum(bool(p[4]['removed']) for p in plans),
                  drop=args.drop, results=[p[4] for p in plans])
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('files', 'matching', 'changed')}))


if __name__ == '__main__':
    main()
