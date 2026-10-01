"""Calibrate AM head budgets using self-study + repeat and answer log-perplexity.

Follows compaction's published head-budget-optimization.sh settings, with SQuAD
and Hugging Face generation. Shared evaluation/AM/patch code is unchanged.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from tqdm import tqdm

from args import parse_ratios
from attention.am import am_compress_head, collect_queries
from attention.am_chunking import fork_reference_chunk_student
from attention.query_cache import fork_generation_cache
from data.load import DATASETS, load_dataset_all
from eval import atomic_json
from patching.references import (
    _AM_QUESTION_PROMPT, _SELF_STUDY_FIXED_PROMPTS, _SELF_STUDY_PREFILL_PROMPT,
    _SELF_STUDY_QUESTION_MAX_TOKENS, _SELF_STUDY_ANSWER_MAX_TOKENS,
    extract_after_thinking_then_split,
)
from utils.func import seed_for_sample


@contextmanager
def preserve_cache(cache):
    """Discard question KV without leaving views into appended storage alive."""
    tensors = [(layer.keys, layer.values) for layer in cache.layers]
    seen = cache._seen_tokens
    try:
        yield
    finally:
        for layer, (keys, values) in zip(cache.layers, tensors):
            layer.keys, layer.values = keys, values
        cache._seen_tokens = seen


@torch.inference_mode()
def prefill_context(model, article):
    """Use AM's closed context turn; retain both non-article prefix and suffix."""
    formatted = model.tokenizer.apply_chat_template([
        dict(role='system', content='You are a helpful assistant. Answer a question based on the provided context.'),
        dict(role='user', content=article),
    ], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    tag = '<|start_header_id|>user' if model.config.model_type == 'llama' else '<|im_start|>user'
    start = formatted.find('\n', formatted.index(tag) + len(tag)) + 1
    # Match compaction/evaluation/utils.py's article token boundaries.
    first = model.encode(formatted[:start]).shape[1]
    last = model.encode(formatted[:start + len(article)]).shape[1]
    ids = model.encode(formatted)
    cache = model._init_kv(evict_range=(first, last))
    cache.ctx_ids, cache.prefill_ids = ids[:, first:last], ids
    cache.calibration_context = formatted
    for chunk in ids.split(4096, dim=1):
        model(chunk, cache, update_cache=True)
    return cache


def format_prompt(model, text, thinking=True):
    return model.tokenizer.apply_chat_template([dict(role='user', content=text)],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


@torch.inference_mode()
def generate_text(model, teacher, question, max_new_tokens, thinking=None):
    """HF sampling with upstream's parameters, without changing model defaults."""
    source = model.model.generation_config
    sampling = dict(temperature=1., top_p=1., top_k=0)
    if thinking is not None:
        sampling.update(temperature=.6 if thinking else .7,
                        top_p=.95 if thinking else .8, top_k=20)
    elif source.do_sample:
        for key in sampling:
            value = getattr(source, key, None)
            if value is not None and (key != 'top_k' or value > 0):
                sampling[key] = value
    config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens,
        bos_token_id=source.bos_token_id, eos_token_id=source.eos_token_id,
        pad_token_id=source.pad_token_id, **sampling)
    inputs = torch.cat([teacher.prefill_ids, question], dim=1)
    cache = fork_generation_cache(teacher, generation_config=config)
    with preserve_cache(teacher):
        tokens = model.model.generate(inputs, past_key_values=cache,
            attention_mask=torch.ones_like(inputs), generation_config=config,
            use_model_defaults=False)
    return model.tokenizer.decode(tokens[0, inputs.shape[1]:], skip_special_tokens=True)


def generate_references(model, teacher, questions, max_new_tokens):
    """Retokenize generated answer text, excluding special tokens such as EOS."""
    answers = []
    for question in tqdm(questions, desc='Full-cache answers', unit='question'):
        text = generate_text(model, teacher, question, max_new_tokens).strip()
        answer = model.encode(text)
        if not answer.shape[1]:
            raise ValueError('Full-cache model generated an empty answer')
        answers.append(answer)
    return answers


def study_sequences(model, teacher):
    """Original ss-plus-repeat prompts/caps, scoped to calibration only."""
    seed_prompt = format_prompt(model, _AM_QUESTION_PROMPT, thinking=True)
    questions = []
    for attempt in range(3):
        response = generate_text(model, teacher, model.encode(seed_prompt),
                                 _SELF_STUDY_QUESTION_MAX_TOKENS, thinking=True)
        questions = extract_after_thinking_then_split(response)
        if questions:
            break
        print(f'Self-study question extraction failed ({attempt + 1}/3)', flush=True)
    article = model.tokenizer.decode(teacher.ctx_ids[0], skip_special_tokens=False)
    specs = [(_SELF_STUDY_PREFILL_PROMPT, False, 0)]
    specs += [(prompt, False, cap) for prompt, cap in _SELF_STUDY_FIXED_PROMPTS]
    specs += [(question, True, _SELF_STUDY_ANSWER_MAX_TOKENS) for question in questions]
    sequences = []
    for prompt, thinking, cap in tqdm(specs, desc='Calibration self-study', unit='reference'):
        prompt = format_prompt(model, prompt, thinking)
        answer = (generate_text(model, teacher, model.encode(prompt), cap, thinking)
                  if cap else article)
        # Qwen2/Llama do not emit thinking tags. Qwen3 thinking must finish.
        if thinking and model.config.model_type == 'qwen3' and '</think>' not in answer:
            print('Skipping unfinished self-study thinking answer', flush=True)
            continue
        full = model.encode(teacher.calibration_context + prompt + answer)
        prefix_len = teacher.prefill_ids.shape[1]
        if not torch.equal(full[:, :prefix_len], teacher.prefill_ids):
            raise ValueError('Self-study tokenization changed the cached context prefix')
        sequences.append(full[:, prefix_len:])
    return sequences


@torch.inference_mode()
def calibration_queries(model, teacher, sequences, max_queries):
    """Sample token positions across attention heads before grouping, as AM does."""
    collect_queries(model, teacher, teacher.ctx_ids, dict(am_query_source='self_study',
        am_prefill_chunk_size=4096, am_max_queries_per_kv_head=0, am_quiet=False),
        self_study_sequences=[(teacher.ctx_ids, ids) for ids in sequences])
    groups, heads = teacher.n_group_kv, teacher.n_heads_kv
    lengths = [chunk.shape[1] for ids in sequences for chunk in ids.split(4096, dim=1)]
    total = sum(lengths)
    count = min(total, math.ceil(max_queries / groups))
    sampled = torch.randperm(total)[:count] if total > count else torch.arange(total)
    kv_sample = (torch.randperm(count * groups, device=model.device)[:max_queries].sort().values
                 if count * groups > max_queries else None)
    queries, teacher.am_queries = teacher.am_queries, None
    teacher.am_query_subsample = None
    for layer, values in enumerate(queries):
        blocks = values.split([length * groups for length in lengths], dim=1)
        grouped = torch.cat([block.reshape(heads, groups, length, -1)
                             for block, length in zip(blocks, lengths)], dim=2)
        values = grouped[:, :, sampled.to(values.device)].reshape(heads, count * groups, -1)
        queries[layer] = values if kv_sample is None else values[:, kv_sample]
    return queries


@torch.inference_mode()
def mean_answer_nll(model, cache, questions, answers, chunk_size=512):
    """Mean per-question log-PPL, scoring only the retokenized answer text."""
    if not questions or len(questions) != len(answers) or chunk_size < 1:
        raise ValueError('Expected paired questions/answers and a positive chunk size')
    losses = []
    for question, answer in zip(questions, answers):
        if question.shape[1] == 0 or answer.shape[1] == 0:
            raise ValueError('Questions and reference answers must be nonempty')
        with preserve_cache(cache):
            # The last prompt token predicts the first answer token. Prefix
            # tokens need decoder states but no vocabulary-sized logits.
            for start in range(0, question.shape[1] - 1, chunk_size):
                ids = question[:, start:min(start + chunk_size, question.shape[1] - 1)]
                model.model.model(ids, past_key_values=cache, use_cache=True)
            inputs = torch.cat([question[:, -1:], answer[:, :-1]], dim=1)
            total = 0.
            for start in range(0, inputs.shape[1], chunk_size):
                ids = inputs[:, start:start + chunk_size]
                logits = model.model(ids, past_key_values=cache, use_cache=True).logits
                targets = answer[:, start:start + ids.shape[1]].to(logits.device)
                total += F.cross_entropy(logits.float().flatten(0, 1),
                                         targets.flatten(), reduction='sum').item()
            losses.append(total / answer.shape[1])
    result = sum(losses) / len(losses)
    if not math.isfinite(result):
        raise FloatingPointError('Non-finite answer log-perplexity')
    return result


@torch.inference_mode()
def replace_head(baseline, teacher, queries, layer_idx, head_idx, budget):
    """Change one head, sharing every other layer of the frozen AM baseline."""
    source = teacher.layers[layer_idx]
    region = slice(teacher.sink, teacher.sink + teacher.ctx_len)
    _, beta, values, indices = am_compress_head(
        queries, source.keys[0, head_idx, region],
        source.values[0, head_idx, region], budget)
    indices, order = indices.sort()
    beta, values = beta[order], values[order]
    cache = copy.copy(baseline)
    cache.layers = [copy.copy(layer) for layer in baseline.layers]
    cache.layers[layer_idx].values = baseline.layers[layer_idx].values.clone()
    cache.layers[layer_idx].values[0, head_idx, teacher.sink + indices] = values
    cache.am_ctx_indices, cache.am_beta = list(baseline.am_ctx_indices), list(baseline.am_beta)
    # SDPA uses a rectangular tensor across heads. Padding has zero mass.
    width = max(baseline.am_ctx_indices[layer_idx].shape[1], len(indices), 1)
    old_indices, old_beta = baseline.am_ctx_indices[layer_idx], baseline.am_beta[layer_idx]
    new_indices = old_indices.new_zeros((baseline.n_heads_kv, width))
    new_beta = old_beta.new_full((1, baseline.n_heads_kv, width), float('-inf'))
    new_indices[:, :old_indices.shape[1]] = old_indices
    new_beta[:, :, :old_beta.shape[2]] = old_beta
    new_indices[head_idx] = 0
    new_beta[0, head_idx] = float('-inf')
    new_indices[head_idx, :len(indices)] = indices
    new_beta[0, head_idx, :len(indices)] = beta.to(new_beta.dtype)
    cache.am_ctx_indices[layer_idx], cache.am_beta[layer_idx] = new_indices, new_beta
    cache.valid = list(baseline.valid)
    cache.valid[layer_idx] = baseline.valid[layer_idx].clone()
    cache.valid[layer_idx][0, head_idx] = False
    cache.valid[layer_idx][0, head_idx, indices] = True
    return cache


def optimize_budget(curves, ratios, step=.001):
    """Greedily swap total-budget shares to reduce mean predicted log-PPL."""
    keys = list(curves)
    if not keys or not ratios or any(not 0 < r <= 1 for r in ratios):
        raise ValueError('Need head curves and target retentions in (0, 1]')
    if not 0 < step < 1:
        raise ValueError('Invalid share step')
    points = [np.asarray(curves[key], dtype=float) for key in keys]
    for curve in points:
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2
                or not np.isfinite(curve).all() or curve[0, 0] != 0
                or not 0 < curve[-1, 0] <= 1 or np.any(np.diff(curve[:, 0]) <= 0)):
            raise ValueError('Curves must start at 0 and have finite, increasing probe points within [0, 1]')
    count = len(keys)
    proportions = np.full(count, 1. / count)
    targets = np.asarray(ratios) * count

    def loss(i, p):
        curve = points[i]
        return float(np.interp(p * targets, curve[:, 0], curve[:, 1]).mean())

    current = np.array([loss(i, p) for i, p in enumerate(proportions)])
    benefit, cost = np.empty(count), np.empty(count)

    def update(i):
        p = proportions[i]
        current[i] = loss(i, p)
        benefit[i] = current[i] - loss(i, p + step) if p + step <= 1 else -np.inf
        cost[i] = loss(i, p - step) - current[i] if p > step else np.inf

    initial = float(current.sum())
    for i in range(count):
        update(i)
    for iteration in range(100000):
        recipient = int(benefit.argmax())
        eligible = cost.copy()
        eligible[recipient] = np.inf
        donor = int(eligible.argmin())
        if benefit[recipient] - eligible[donor] <= 0:
            break
        proportions[recipient] += step
        proportions[donor] -= step
        update(recipient)
        update(donor)
    else:
        raise RuntimeError('Budget solver did not converge in 100000 swaps')
    proportions /= proportions.sum()
    stats = dict(swaps=iteration, uniform_predicted_delta=initial,
                 optimized_predicted_delta=float(current.sum()))
    return dict(zip(keys, proportions.tolist())), stats


@torch.inference_mode()
def calibrate_context(model, record, questions, args, path, identity):
    """Checkpoint after each head; regenerate neither saved answers nor curves."""
    signature = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    metadata = dict(identity=identity, context_sha256=signature,
                    question_ids=[q[0].tolist() for q in questions])
    payload = json.loads(path.read_text()) if path.exists() else dict(metadata=metadata, curves={})
    if payload['metadata'] != metadata:
        raise ValueError(f'Calibration settings/data differ from saved run: {path}')
    keys = [f'L{l}H{h}' for l in range(model.config.num_hidden_layers)
            for h in range(model.config.num_key_value_heads)]
    if set(payload['curves']) == set(keys):
        print(f'Skip completed {path.name}', flush=True)
        return payload['curves']
    seed_for_sample(args.seed, 0)
    teacher = prefill_context(model, record['context'])
    if teacher.ctx_len == 0:
        raise ValueError('Calibration context is empty')
    if 'answers' not in payload:
        answers = generate_references(model, teacher, questions, args.max_new_tokens)
        payload['answers'] = [a[0].tolist() for a in answers]
        payload['context_tokens'] = teacher.ctx_len
        atomic_json(path, payload)
    answers = [torch.tensor([a], device=model.device) for a in payload['answers']]
    if 'references' not in payload:
        seed_for_sample(args.seed, 1)
        sequences = study_sequences(model, teacher)
        payload['references'] = [ids[0].tolist() for ids in sequences]
        atomic_json(path, payload)
    sequences = [torch.tensor([ids], device=model.device) for ids in payload['references']]
    # Save generated reference tokens too: resumed probes use identical queries.
    seed_for_sample(args.seed, 1)
    queries = calibration_queries(model, teacher, sequences, args.max_queries)
    baseline = fork_reference_chunk_student(teacher)
    # Match the upstream integer total budget and minimum of one key per head.
    total_budget = int(args.reference_ratio * teacher.ctx_len * len(keys))
    baseline_tokens = max(1, int(total_budget / len(keys)))
    baseline.am_head_budget = None
    baseline.prune_am(min(1., (baseline_tokens + .5) / teacher.ctx_len), queries, verbose=False)
    baseline_nll = mean_answer_nll(model, baseline, questions, answers)
    payload['baseline_log_ppl'] = baseline_nll
    payload['queries_per_kv_head'] = queries[0].shape[1]
    grid = np.linspace(0, args.max_ratio, args.n_points).tolist()
    for layer in range(teacher.n_layers):
        for head in range(teacher.n_heads_kv):
            key = f'L{layer}H{head}'
            if key in payload['curves']:
                continue
            # Different probe ratios can round to the same integer budget.
            measured = {}
            curve = []
            for ratio in grid:
                budget = max(1, int(ratio * teacher.ctx_len))
                if budget not in measured:
                    seed_for_sample(args.seed, 2 + (layer * teacher.n_heads_kv + head) *
                                    (teacher.ctx_len + 1) + budget)
                    changed = replace_head(baseline, teacher, queries[layer][head], layer, head, budget)
                    measured[budget] = mean_answer_nll(model, changed, questions, answers) - baseline_nll
                    del changed
                curve.append([ratio, measured[budget]])
            payload['curves'][key] = curve
            atomic_json(path, payload)
            print(f'{path.stem}: {len(payload["curves"])}/{len(keys)} heads', flush=True)
    return payload['curves']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model', default='qwen3-4b')
    parser.add_argument('-d', '--data', choices=DATASETS, default='squad')
    parser.add_argument('--idx', type=int, default=100, help='First calibration context; keep disjoint from evaluation')
    parser.add_argument('--num', type=int, default=10, help='Contexts; use all their questions')
    parser.add_argument('--n_points', type=int, default=51, help='Evenly spaced probes from 0 to --max_ratio')
    parser.add_argument('--max_ratio', type=float, default=.5, help='Largest per-head probe retention')
    parser.add_argument('--reference_ratio', type=float, default=.05)
    parser.add_argument('--ratios', type=parse_ratios, default=parse_ratios('.01,.02,.05,.1'))
    parser.add_argument('--max_queries', type=int, default=50000, help='Collected AM queries per KV head')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='Full-cache answer generation cap')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True, help='Output budget JSON; intermediates use <stem>.calibration/')
    args = parser.parse_args(argv)
    if args.idx < 0 or args.n_points < 2 or any(getattr(args, k) < 1 for k in (
            'num', 'max_queries', 'max_new_tokens')):
        parser.error('Use nonnegative --idx, --n_points >= 2 and positive counts')
    if not 0 < args.reference_ratio < 1 or any(r == 0 for r in args.ratios):
        parser.error('Reference retention must be in (0, 1); target retentions must be positive')
    if not 0 < args.max_ratio <= 1:
        parser.error('--max_ratio must be in (0, 1]')
    if args.output.suffix != '.json':
        parser.error('--output must end in .json')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('AM budget calibration requires CUDA; no model was loaded.')
    from model import ModelKVzip
    run_calibration(args, ModelKVzip(args.model))


def run_calibration(args, model):
    data = load_dataset_all(args.data, model.tokenizer, n_data=args.idx + args.num)
    if args.idx + args.num > len(data):
        raise ValueError(f'Requested contexts exceed dataset size {len(data)}')
    workdir = args.output.with_suffix('.calibration')
    settings = {k: v for k, v in vars(args).items() if k != 'output'}
    identity = dict(schema='am-budget-ss-plus-repeat-v2', settings=settings,
        model_config=model.config.to_dict(), model_dtype=str(model.dtype),
        generation_config=model.model.generation_config.to_dict(),
        source_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ('scripts/calibrate_am_budget.py', 'attention/am.py', 'attention/am_cache.py',
                         'attention/attn.py', 'model/wrapper.py', 'model/load.py',
                         'patching/references.py', 'data/load.py')})
    # Canonicalize config values (e.g. integer dict keys) before resume checks.
    identity = json.loads(json.dumps(identity))
    manifest = workdir / 'settings.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(f'Use a different --output for changed settings: {manifest}')
    elif args.output.exists():
        raise FileExistsError(f'Refusing to replace a budget without its calibration records: {args.output}')
    atomic_json(manifest, identity)
    all_curves = []
    for idx in range(args.idx, args.idx + args.num):
        path = workdir / f'context_{idx:05d}.json'
        questions = [model.encode(format_prompt(model, question)) for question in data[idx]['question']]
        all_curves.append(calibrate_context(model, data[idx], questions, args, path, identity))
    curves = {key: np.mean([c[key] for c in all_curves], axis=0).tolist() for key in all_curves[0]}
    budget, stats = optimize_budget(curves, args.ratios)
    atomic_json(workdir / 'aggregate.json', dict(curves=curves, solver=stats))
    atomic_json(args.output, budget)
    print(f'Budget saved: {args.output}\nSolver: {stats}', flush=True)


if __name__ == '__main__':
    main()
