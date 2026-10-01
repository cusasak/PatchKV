import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from results.metric import evaluate_answer, strip_thinking


def parse_run_dir(path):
    path = Path(path)
    match = re.fullmatch(
        r'(.+)_(kvzip|am)_(unpatched|repeat-context|self_study|joint)_'
        r'[0-9]{8}_[0-9]{6}(?:_[0-9]+)?', path.name)
    if match is None:
        raise ValueError(f'Keep QA files in their original model/run folders: {path}')
    dataset, baseline, proxy = match.groups()
    return dict(model=path.parent.name, data=dataset, baseline=baseline,
                patch=proxy != 'unpatched', patch_data=proxy, run=path.name)


def load_context(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {'questions', 'answers', 'results'}:
        raise ValueError(f'Unsupported or empty QA result: {path}')
    return dict(value, config=parse_run_dir(path.parent),
                context_index=int(path.stem.removeprefix('context_')))


def summarize(paths):
    groups = defaultdict(list)
    run_ratios = {}
    repoqa_data = {}
    seen = set()
    for path in paths:
        value = load_context(path)
        config = value['config']
        # Separate runs must never silently contribute to the same mean.
        protocol = json.dumps(config, sort_keys=True)
        identity = protocol, value['context_index']
        if identity in seen:
            raise ValueError(f'Duplicate context: {path}')
        seen.add(identity)
        actual = [row['retention'] for row in value['results']]
        expected = set(actual)
        if (1. not in expected or len(actual) != len(expected)
                or any(not 0 <= ratio <= 1 for ratio in actual)):
            raise ValueError(f'Missing or duplicate retention results: {path}')
        if protocol in run_ratios and run_ratios[protocol] != expected:
            raise ValueError(f'Retention results differ within run: {path}')
        run_ratios[protocol] = expected
        if not value['questions'] or len(value['answers']) != len(value['questions']):
            raise ValueError(f'Question/answer count differs: {path}')
        references = value['answers']
        if 'repoqa' in config['data']:
            from data.load import load_scbench
            if config['data'] not in repoqa_data:
                repoqa_data[config['data']] = load_scbench(config['data'])
            record = repoqa_data[config['data']][value['context_index']]
            if record['question'] != value['questions'] or record['answers'] != value['answers']:
                raise ValueError(f'RepoQA source differs from saved QA: {path}')
            references = record['repoqa_refs']
        for row in value['results']:
            if len(row['predictions']) != len(value['questions']):
                raise ValueError(f'Question count differs: {path}')
            scores = evaluate_answer([strip_thinking(p) for p in row['predictions']],
                                     references, config['data'], 'qa')
            score = sum(scores) / len(scores)
            groups[protocol, row['retention']].append(score)
    rows = []
    for (protocol, retention), scores in sorted(
            groups.items(), key=lambda item: (item[0][0], -item[0][1])):
        config = json.loads(protocol)
        rows.append(dict(model=config['model'], dataset=config['data'], baseline=config['baseline'],
                         patch=config['patch_data'] if config['patch'] else 'none',
                         retention=retention, contexts=len(scores), score=100 * sum(scores)/len(scores),
                         run=config['run']))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', help='Run directories or context JSON files')
    args = parser.parse_args(argv)
    paths = []
    for name in args.paths:
        path = Path(name)
        paths.extend(sorted(path.rglob('context_*.json')) if path.is_dir() else [path])
    if not paths:
        parser.error('No context result files found')
    last_run = None
    for row in summarize(paths):
        run = row['model'], row['run']
        if run != last_run:
            print(f"\n{row['model']} / {row['run']}")
            print('retention  contexts  score (%)')
            last_run = run
        print(f"{row['retention']:9g}  {row['contexts']:8d}  {row['score']:9.2f}")


if __name__ == '__main__':
    main()
