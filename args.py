import argparse
import math
from data.load import DATASETS


def parse_ratios(value):
    try:
        ratios = [float(x) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated retention ratios') from exc
    if not ratios or any(not 0 <= x <= 1 for x in ratios) or len(set(ratios)) != len(ratios):
        raise argparse.ArgumentTypeError('Ratios must be unique values in [0, 1]')
    return ratios


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate KVzip or AM with PatchKV')
    parser.add_argument('-m', '--model', default='qwen2.5-7b')
    parser.add_argument('-d', '--data', choices=DATASETS, default='squad')
    parser.add_argument('--baseline', choices=['kvzip', 'am'], default='kvzip')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--patch_data', choices=['repeat-context', 'self_study', 'joint'], default='repeat-context',
                        help='joint concatenates repeat-context and self-study references')
    parser.add_argument('--chunk_length', type=int, default=5000, help='Repeat patch chunk size')
    parser.add_argument('--lam', type=float, default=.001, help='Weighted ridge coefficient')
    parser.add_argument('--level', choices=['pair', 'pair-uniform'], default='pair')
    parser.add_argument('--repeat_chunk_size', type=int, default=2000, help='KVzip scoring chunk size')
    parser.add_argument('--prefill_chunk_size', type=int, default=16000)
    parser.add_argument('--ratios', type=parse_ratios, default=parse_ratios('0.2,0.1,0.05,0.02,0.01,0'),
                        help='Comma-separated retentions; full-cache control is always evaluated')
    parser.add_argument('--idx', type=int, default=0, help='First context index')
    parser.add_argument('--num', type=int, default=100, help='Number of contexts, all their questions')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enable_thinking', action='store_true', help='Enable Qwen3 thinking for evaluation')
    parser.add_argument('--max_new_tokens', type=int, help='Override evaluation cap, not self-study caps')
    parser.add_argument('--am_on_policy', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--am_chunk_size', type=int, default=0,
                        help='AM article chunks with separate references (0: entire context)')
    parser.add_argument('--am_prefill_chunk_size', type=int, default=4096)
    parser.add_argument('--am_max_queries_per_kv_head', type=int, default=50000)
    parser.add_argument('--am_head_budget', help='Optional per-head budget JSON from AM')
    parser.add_argument('--am_query_source', choices=['self_study', 'repeat'], default='self_study')
    parser.add_argument('--include_repeat_prefill', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--result_root', default='results')
    parser.add_argument('--resume', help='Existing run directory to resume with the same evaluation options')
    args = parser.parse_args(argv)
    for name in ('chunk_length', 'repeat_chunk_size', 'prefill_chunk_size', 'num', 'am_prefill_chunk_size', 'am_max_queries_per_kv_head'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.idx < 0 or args.am_chunk_size < 0:
        parser.error('--idx and --am_chunk_size must be nonnegative')
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        parser.error('--max_new_tokens must be positive')
    if not math.isfinite(args.lam) or args.lam < 0:
        parser.error('--lam must be finite and nonnegative')
    return args
