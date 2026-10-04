"""Read-only checks of saved smoke evidence. No training and no data download."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def verify(folder):
    tasks=json.loads((folder/'tasks.json').read_text())
    with (folder/'metrics.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    successes=[t for t in tasks if t['exit_code']==0]
    assert len(rows)==len(successes),(folder,len(rows),len(successes))
    for task in successes:
        # Resolve relative to the result tree, so the evidence can be moved.
        out=folder/task['method']/task['dataset']/f'seed_{task["seed"]}'
        status=json.loads((out/'status.json').read_text())
        assert status['status']=='success'
        assert (out/'model.pt').stat().st_size==status['model_bytes']
        total=0
        for path in [out/'prepare_flops.json',*sorted(out.glob('epoch_*_operations.json'))]:
            report=json.loads(path.read_text())
            assert report['complete'] and not report['unknown'],path
            assert report['flops']==sum(r['flops'] for r in report['operations']),path
            assert report['fft_flops_estimated']==sum(r['fft_flops_estimated'] for r in report['operations']),path
            if 'known_flops_by_stage' in report:
                assert report['flops']==sum(report['known_flops_by_stage'].values()),path
            for row in report['operations']:
                assert row['flops']==row['calls']*row['flops_per_call'],path
            total+=report['flops']
        assert total==status['total_flops'],out
    for row in rows:
        if row.get('epoch_flops'):
            assert int(row['epoch_flops'])==int(row['train_flops'])+int(row['test_flops'])
    return dict(folder=str(folder.relative_to(ROOT)),tasks=len(tasks),successes=len(successes),failures=len(tasks)-len(successes))


if __name__=='__main__':
    for group in ('implementation_final_cpu','implementation_final_gpu','implementation_full_time',
                  'implementation_shd','implementation_failure_continue'):
        folders=sorted((ROOT/group).glob('run_*'))
        assert folders,group
        print(json.dumps(verify(folders[-1]),ensure_ascii=False))
