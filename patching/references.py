import re
from typing import List, Tuple
import torch
from attention.query_cache import fork_generation_cache

_SELF_STUDY_FIXED_PROMPTS = [
    ("Summarize the main points of the context.", 2048),
    ("Aggregate all the key facts mentioned in the context.", 2048),
    ("Structure the information in JSON form and include all important details like dates, times, names, and numerical values.", 2048),
]


_PATCH_QUESTION_PROMPT = (
    "Write 3 questions that test understanding of different parts of the context. "
    "Answer with just the 3 questions, each separated with 2 newlines."
)


_AM_QUESTION_PROMPT = (
    "Write 3 questions that test understanding of different parts of the context. "
    "Answer with just the 3 questions and options (do not say the correct answer), "
    "each one separated with 2 newlines."
)


_SELF_STUDY_QUESTION_MAX_TOKENS = 8192


_SELF_STUDY_ANSWER_MAX_TOKENS = 4096


_SELF_STUDY_PREFILL_PROMPT = "Repeat the previous context verbatim."


_SELF_STUDY_FIXED_SPEC_NAMES = ('summarize', 'aggregate', 'structure_json')


def self_study(
    model,
    kv,
    ctx_ids: torch.Tensor,
    question_prompt: str = _PATCH_QUESTION_PROMPT,
    include_prefill: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    results = []
    seen_token_prev = kv._seen_tokens

    def _generate_response(prompt_text, per_spec_max_tokens, label=""):
        gen_kwargs = dict(model.gen_kwargs, max_new_tokens=int(per_spec_max_tokens))
        q_ids = model.encode(f"\n\n{prompt_text}")
        query_ids = torch.cat([q_ids, model.self_study_postfix_ids], dim=1)
        full_input_ids = torch.cat([kv.prefill_ids, query_ids], dim=1)
        generation_kv = fork_generation_cache(
            kv, getattr(model.model, "generation_config", None), **gen_kwargs
        )
        try:
            output = model.model.generate(
                full_input_ids, past_key_values=generation_kv, **gen_kwargs
            )
        finally:
            # Plain RetainCache shares the immutable context and keeps only
            # this prompt's new KV. Other cache modes retain legacy cleanup.
            if generation_kv is kv:
                kv.slice(seen_token_prev)
        response_ids = output[:, len(full_input_ids[0]):-1]
        response_text = model.decode(response_ids)
        if label:
            print(f"\n  {label}")
            print(f"  Q: {prompt_text[:120]}")
            print(f"  A: {response_text[:300]}")
        return query_ids, response_ids, response_text

    print("\n" + "-" * 60)

    if include_prefill:
        # "repeat-prefill" entry: skip model.generate, use ctx_ids as the answer.
        # No thinking involved (article tokens are the "answer").
        prefill_q = model.encode(f"\n\n{_SELF_STUDY_PREFILL_PROMPT}")
        prefill_query_ids = torch.cat([prefill_q, model.self_study_postfix_ids], dim=1)
        prefill_study_ids = torch.cat([prefill_query_ids, ctx_ids], dim=1)
        results.append((ctx_ids, prefill_study_ids))
        print(f"[Self-study] repeat-prefill: q={prefill_query_ids.shape[1]} tok, "
              f"a={ctx_ids.shape[1]} tok (article tokens, no generation)")

    # Step 1: summary, key facts, and structured JSON.
    fixed_specs = [
        (name, prompt, mt)
        for name, (prompt, mt) in zip(
            _SELF_STUDY_FIXED_SPEC_NAMES, _SELF_STUDY_FIXED_PROMPTS,
        )
    ]
    print(f"[Self-study] Step 1: {len(fixed_specs)} fixed prompts")

    for p_idx, (spec_name, prompt_text, spec_max_tokens) in enumerate(fixed_specs):
        label = f"[Fixed {p_idx+1}/{len(fixed_specs)} :: {spec_name}]"
        query_ids, response_ids, _ = _generate_response(
            prompt_text, spec_max_tokens, label=label,
        )
        study_ids = torch.cat([query_ids, response_ids], dim=1)
        results.append((ctx_ids, study_ids))

    # Step 2: generate distinct questions and answer each once.
    print("\n[Self-study] Step 2: generating questions from context...")
    _, _, seed_resp_text = _generate_response(
        question_prompt,
        _SELF_STUDY_QUESTION_MAX_TOKENS,
        label="[Question gen]",
    )
    # Keep each generated question together with all of its option lines.
    # The extractor also removes Model A's thinking portion when present.
    questions = self_study_questions(seed_resp_text)
    if len(questions) < 3:
        print(f"[Self-study] Extracted {len(questions)} distinct questions; "
              "using only those questions without duplicating the seed response.")

    print(f"[Self-study] Step 2: answering {len(questions)} generated questions")
    for q_idx, question in enumerate(questions):
        label = f"[Generated q{q_idx+1}/{len(questions)}]"
        query_ids, response_ids, _ = _generate_response(
            question,
            _SELF_STUDY_ANSWER_MAX_TOKENS,
            label=label,
        )
        study_ids = torch.cat([query_ids, response_ids], dim=1)
        results.append((ctx_ids, study_ids))

    print("-" * 60 + "\n")

    return results


def extract_after_thinking_then_split(text: str) -> List[str]:
    """Extract question blocks while keeping each question's options together.

    This mirrors compaction's ``3_question`` extraction behavior. It accepts
    common model formats: separator lines, numbered questions, double-newline
    blocks, and separate question/options block pairs.
    """
    if "</think>" in text:
        content = text.split("</think>", 1)[1].strip()
    else:
        content = text.strip()

    if not content:
        return []

    # Prefer explicit horizontal separator lines.
    items = re.split(r"\n+\s*-{3,}\s*\n+", content)

    # Numbered question boundaries keep all following option lines attached.
    if len(items) == 1:
        numbered_items = re.split(r"\n+(?=\d+[\.\)]\s)", content)
        if len(numbered_items) > 1:
            items = numbered_items

    # The prompt asks for each question block to be separated by a blank line.
    if len(items) == 1:
        items = re.split(r"\n\s*\n", content)

    # Free-form questions sometimes arrive as one question per line, including
    # Markdown hard breaks (two spaces before a newline). Do not split option
    # lines or wrapped stems: every line must itself be a complete question.
    if len(items) == 1:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) > 1 and all(line.endswith("?") for line in lines) and not any(
            re.match(r"^[A-H][\).]\s", line) for line in lines
        ):
            items = lines

    items = [item.strip() for item in items if item.strip()]

    # Some models insert a blank line between a stem and its options, producing
    # [question, options, question, options, ...]. Rejoin those pairs.
    if len(items) > 5:
        options_pattern = re.compile(r"^A[\)\.]")
        paired_options = all(
            options_pattern.match(items[i])
            for i in range(1, len(items), 2)
        )
        if paired_options and len(items) % 2 == 0:
            items = [
                items[i] + "\n\n" + items[i + 1]
                for i in range(0, len(items), 2)
            ]

    # Match the reference's defensive rejection of badly fragmented output.
    if len(items) > 5:
        return []
    return items


def self_study_questions(text: str) -> List[str]:
    """Select up to three distinct questions; never pad with the seed response.

    If the model emits only one question, answer it once. Fixed self-study
    tasks still provide reference sequences when no questions are extracted.
    """
    questions = []
    seen = set()
    for question in extract_after_thinking_then_split(text):
        key = " ".join(question.split())
        if key and key not in seen:
            seen.add(key)
            questions.append(question)
        if len(questions) == 3:
            break
    return questions


def am_references(model, teacher, include_prefill=True):
    """Generate AM's MCQ references once from the uncompressed teacher."""
    return self_study(model, teacher, teacher.ctx_ids,
                      question_prompt=_AM_QUESTION_PROMPT, include_prefill=include_prefill)


def patch_references(model, teacher, source, chunk_length=5000):
    """Build repeat-context, self-study, or joint references for one context."""
    if source not in {'repeat-context', 'self_study', 'joint'}:
        raise ValueError(f'Unknown patch reference: {source}')
    sequences = []
    if source in {'repeat-context', 'joint'}:
        sequences.extend(model.self_task(teacher.ctx_ids, chunk_size=chunk_length)[0])
    if source in {'self_study', 'joint'}:
        pairs = self_study(model, teacher, teacher.ctx_ids)
        sequences.extend((teacher.ctx_ids, study_ids, True) for _, study_ids in pairs)
    return sequences
