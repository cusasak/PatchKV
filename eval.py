import json
import os
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

from args import parse_args


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_output_dir(args, model_name):
    variant = args.patch_data if args.patch else 'unpatched'
    if args.resume:
        from results.parse import parse_run_dir
        path = Path(args.resume).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'Resume directory does not exist: {path}')
        config = parse_run_dir(path)
        if (config['model'], config['data'], config['baseline'], config['patch_data']) != (
                model_name, args.data, args.baseline, variant):
            raise ValueError(f'Resume directory does not match the model, dataset, or method: {path}')
        expected = {1., *args.ratios}
        for saved in sorted(path.glob('context_*.json')):
            value = json.loads(saved.read_text())
            ratios = [row['retention'] for row in value['results']]
            if set(ratios) != expected or len(ratios) != len(expected):
                raise ValueError(f'Resume retention ratios differ from saved results: {saved}')
        return path

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    name = f'{args.data}_{args.baseline}_{variant}_{timestamp}'
    for serial in count():
        suffix = f'_{serial}' if serial else ''
        path = Path(args.result_root) / model_name / f'{name}{suffix}'
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path


def score_predictions(predictions, record, dataset):
    from results.metric import evaluate_answer, strip_thinking
    references = record.get('repoqa_refs', record['answers'])
    scores = evaluate_answer([strip_thinking(p) for p in predictions], references, dataset, 'qa')
    if not scores:
        raise ValueError('Cannot score a context with no questions')
    return list(map(float, scores)), sum(scores) / len(scores)


def main(argv=None):
    args = parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Model evaluation requires CUDA.')
    from model import ModelKVzip
    from utils.func import seed_for_sample
    seed_for_sample(args.seed, 0)
    run_evaluation(args, ModelKVzip(args.model))


def run_evaluation(args, model):
    import torch
    from attention.kvcache import fork_cache
    from data.load import load_dataset_all
    from data.wrapper import DataWrapper
    from utils.func import set_gen_length, seed_for_sample
    from attention.am_budget import load_head_budget
    from attention.am import AMPreparation

    raw_data = load_dataset_all(args.data, model.tokenizer, n_data=max(100, args.idx + args.num))
    dataset = DataWrapper(args.data, raw_data, model, enable_thinking=args.enable_thinking)
    if args.idx >= len(dataset):
        raise ValueError(f'Context index {args.idx} exceeds dataset size {len(dataset)}')
    cap = set_gen_length(args.data, model)
    model.gen_kwargs['max_new_tokens'] = args.max_new_tokens or (max(cap, 2048) if args.enable_thinking else cap)
    patcher = None
    if args.patch:
        from patching import PatchKV, patch_references
        patcher = PatchKV(model)
    head_budget = load_head_budget(args.am_head_budget, model.config.num_hidden_layers,
                                   model.config.num_key_value_heads) if args.am_head_budget else None
    output_dir = prepare_output_dir(args, model.name)
    print(f'Results: {output_dir}', flush=True)

    for idx in range(args.idx, min(len(dataset), args.idx + args.num)):
        path = output_dir / f'context_{idx:05d}.json'
        record = raw_data[idx]
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous['questions'] != record['question']
                    or previous['answers'] != record['answers']):
                raise ValueError(f'Saved questions or answers differ: {path}')
            saved_ratios = [row['retention'] for row in previous['results']]
            expected_ratios = {1., *args.ratios}
            if (set(saved_ratios) == expected_ratios
                    and len(saved_ratios) == len(expected_ratios)
                    and all(len(row['predictions']) == len(record['question'])
                            for row in previous['results'])):
                print(f'Skip completed context {idx}', flush=True)
                continue
        seed_for_sample(args.seed, idx)
        record = raw_data[idx]
        teacher = model.prefill(
            record['context'], prefill_chunk_size=args.prefill_chunk_size,
            do_score=args.baseline == 'kvzip',
            score_config=dict(repeat_chunk_size=args.repeat_chunk_size))
        queries = dataset.queries(idx)
        print(f'context={idx} full-cache evaluation start', flush=True)
        full_predictions = dataset.generate_answers(queries, teacher)
        # Match KVzip's saved full-answer token round trip.
        full_predictions = [model.decode(model.encode(text)) for text in full_predictions]
        _, full_score = score_predictions(full_predictions, record, args.data)
        payload = dict(questions=record['question'], answers=record['answers'],
                       results=[dict(retention=1., predictions=full_predictions)])
        print(f'context={idx} full-cache score={100 * full_score:.2f}%', flush=True)

        compressed_ratios = [ratio for ratio in args.ratios if ratio != 1.]
        # Generate all reference tokens from the original full-cache model.
        sequences = []
        if args.patch and compressed_ratios:
            print(f'context={idx} patch references start ({args.patch_data})', flush=True)
            sequences = patch_references(model, teacher, args.patch_data, args.chunk_length)
            print(f'context={idx} patch references done', flush=True)
        am = None
        if args.baseline == 'am' and compressed_ratios:
            print(f'context={idx} AM preparation start', flush=True)
            am = AMPreparation(model, teacher, args, head_budget)
            print(f'context={idx} AM preparation done', flush=True)
        for ratio in compressed_ratios:
            status = f'context={idx} ratio={ratio:g}'
            try:
                print(f'{status} {args.baseline} compression start', flush=True)
                if am is not None:
                    student, _ = am.compress(ratio)
                else:
                    student = fork_cache(teacher)
                    student.prune(ratio, args.level)
                print(f'{status} {args.baseline} compression done', flush=True)
                if args.patch:
                    print(f'{status} patch start', flush=True)
                    patcher.apply(teacher, student, sequences, lam=args.lam)
                    print(f'{status} patch done', flush=True)
                print(f'{status} evaluation start', flush=True)
                predictions = dataset.generate_answers(queries, student)
                _, score = score_predictions(predictions, record, args.data)
                payload['results'].append(dict(retention=ratio, predictions=predictions))
                print(f'context={idx} ratio={ratio:g} score={100 * score:.2f}%', flush=True)
            finally:
                if args.patch:
                    patcher.reset()
            del student
        # A file appears only when all retentions finish; no status file is needed.
        atomic_json(path, payload)
        del teacher, queries, sequences, am
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
    print(f'Complete. Aggregate with: python -m results.parse {output_dir}', flush=True)
    return output_dir


if __name__ == '__main__':
    main()
