"""Audit native-entry results, including incomplete FLOPs and preserved failures."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pp3.native_runtime import check_finite_state

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
report = []
for launch in sorted(args.directory.rglob('launch.json')):
    config = json.loads(launch.read_text(encoding='utf-8'))
    root = launch.parent
    status = json.loads((root/'status.json').read_text(encoding='utf-8'))
    unknown, phases = set(), 0
    for path in root.glob('*_operations.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        assert sum(op['flops'] for op in result['operations']) == result['known_flops'], path
        assert sum(result['known_flops_by_stage'].values()) == result['known_flops'], path
        assert result['complete'] == (not result['unknown']), path
        assert result['flops'] == (result['known_flops'] if result['complete'] else None), path
        for op in result['unknown']:
            unknown.add(op['operator']+': '+op['rule'])
        for key in ('accuracy', 'loss'):
            assert result.get(key) is None or math.isfinite(result[key]), (path, key)
        phases += 1
    metrics = root/'metrics.csv'
    rows = list(csv.DictReader(metrics.open(encoding='utf-8', newline=''))) if metrics.exists() else []
    if status.get('training_completed'):
        assert (root/'model.pt').stat().st_size == status['model_bytes'], root
        check_finite_state(torch.load(root/'model.pt', map_location='cpu', weights_only=False))
        assert rows, root
        assert all(int(row['final_model_bytes']) == status['model_bytes'] for row in rows), root
    report.append(dict(method=config['method'], dataset=config['dataset'], seed=config['seed'],
                       entrypoint=config['entrypoint'], status=status['status'], phases=phases,
                       rows=len(rows), unknown=sorted(unknown), output=str(root)))
destination = args.directory/'native_audit.json'
destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps([dict(method=r['method'],dataset=r['dataset'],status=r['status'],
                       unknown_count=len(r['unknown'])) for r in report], indent=2))
print(destination)
