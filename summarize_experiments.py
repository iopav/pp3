"""Aggregate completed experiments across jobs. Does not import torch or train."""
import argparse
import json
from pathlib import Path
from pp3.completed import completed_status
from pp3.reporting import export_summary


def summarize(directories, destination):
    candidates={}; excluded=[]; selected={}
    for directory in directories:
        root=Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for launch in root.rglob('launch.json'):
            if '_history' in launch.relative_to(root).parts:
                continue
            candidates[str(launch.parent)]=launch.parent
        # Manifests may point to already-completed runs skipped by a later job.
        for manifest in root.rglob('tasks.json'):
            if '_history' in manifest.relative_to(root).parts:
                continue
            try:
                for task in json.loads(manifest.read_text(encoding='utf-8')):
                    out=Path(task['output']).resolve()
                    if out.is_dir(): candidates[str(out)]=out
            except (OSError,ValueError,KeyError,TypeError):
                excluded.append(dict(path=str(manifest),reason='Unreadable task manifest'))
    for out in sorted(candidates.values()):
        try:
            config=json.loads((out/'launch.json').read_text(encoding='utf-8'))
            status=completed_status(out)
            if not status:
                excluded.append(dict(path=str(out),reason='Not completed or required artifacts invalid'))
                continue
            # Older runs lack a source/config identity; preserve them separately.
            key=config.get('experiment_id') or 'legacy:'+str(out)
            task={k:config[k] for k in ('method','dataset','seed')}
            task.update(output=str(out),experiment_id=key)
            stamp=(out/'status.json').stat().st_mtime_ns
            if key not in selected or stamp>selected[key][0]:
                selected[key]=(stamp,task)
        except (OSError,ValueError,KeyError,TypeError):
            excluded.append(dict(path=str(out),reason='Unreadable experiment configuration'))
    tasks=[value[1] for _,value in sorted(selected.items())]
    destination=Path(destination).expanduser().resolve()
    export_summary(destination.parent,tasks=tasks,destination=destination)
    destination.with_suffix('.selection.json').write_text(json.dumps(dict(
        input_directories=list(map(str,directories)),selected_tasks=tasks,excluded=excluded,
        note='FLOPs column uses counted operations even when incomplete. Coverage and missing rules remain in each task operations JSON/status.json; values are not certified full totals.'
    ),indent=2),encoding='utf-8')
    print(f'Completed experiments: {len(tasks)}; excluded artifacts: {len(excluded)}')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',default=['result'])
    parser.add_argument('--output',default='result/summary.csv')
    args=parser.parse_args()
    print(summarize(args.inputs,args.output))
