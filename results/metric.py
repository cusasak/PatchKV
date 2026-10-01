# Copyright (c) 2024 Microsoft; licensed under MIT.
# Adapted from microsoft/MInference/scbench and KVzip; see LICENSE.
import re
import string
from collections import Counter, defaultdict
from rouge import Rouge
from results.repo_qa_utils import compute_score as compute_repoqa_score

_THINK_CLOSE_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_DANGLING_RE = re.compile(r"^.*?</think>", re.DOTALL)

def strip_thinking(s):
    """Remove Qwen3 ``<think>...</think>`` chain-of-thought from a prediction.

    Handles three cases:
      - balanced ``<think>..</think>`` anywhere: drop the whole block.
      - dangling ``</think>`` (max_new_tokens cut the open tag): drop everything
        up to and including the first ``</think>``.
      - no think tags: return as-is.
    """
    if not isinstance(s, str):
        return s
    if "<think>" in s or "</think>" in s:
        s = _THINK_CLOSE_RE.sub("", s)
        if "</think>" in s:
            s = _THINK_DANGLING_RE.sub("", s, count=1)
    return s.strip()

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # Strip Qwen3 thinking traces first (no-op for non-thinking outputs).
    s = strip_thinking(s)

    def replace_num(text):
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

        pattern = re.compile(r'\b(' + '|'.join(word_to_number.keys()) + r')\b')
        text = pattern.sub(lambda x: word_to_number[x.group()], text)

        return text

    return replace_num(white_space_fix(remove_articles(remove_punc(lower(s)))))

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]

def f1_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    prediction_tokens = pred.split()
    ground_truth_tokens = ref.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def include_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred

def include_score_gsm(pred, ref, normalize=True):
    ref = ref.strip().split("#### ")[-1]
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return ref in pred

def exact_match_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return pred == ref

def repoqa_score(preds, refs, subtask=None):
    needle_by_repo = defaultdict(list)
    for name, gt in zip(refs["func_name"], refs["ground_truth"]):
        needle_by_repo[refs["repo"]].append({"needle": gt, "name": name})

    pred_list = []

    for idx in range(len(preds)):
        if subtask is not None:
            if not "repoqa" in subtask[idx]:
                continue

        result = {}
        result["prediction"] = preds[idx]
        if preds[idx].endswith("</s>"):
            result["prediction"] = preds[idx][:-4]
        if len(result["prediction"].strip()) == 0:
            continue

        result["lang"] = refs["lang"]
        result["repo"] = refs["repo"]
        result["func_name"] = refs["func_name"][idx]
        result["ground_truth"] = refs["ground_truth"][idx]
        pred_list.append(result)

    if not pred_list:
        return 0.0
    acc = compute_repoqa_score(pred_list, None, needle_by_repo)
    acc = acc["scores"]["all"][0.8]["pass@1"]
    return acc * len(pred_list) / len(preds) if preds else 0.0


def evaluate_answer(preds, refs, dataname, format="qa", similarity=False, subtask=None):
    if "repoqa" in dataname and not similarity:
        if len(preds) != len(refs["ground_truth"]):
            raise ValueError("RepoQA prediction/reference count differs")
        return [repoqa_score(preds, refs)]
    if len(preds) != len(refs):
        raise ValueError("Prediction/reference count differs")
    scores = []
    for pred, ref in zip(preds, refs):
        if pred.endswith("</s>"):
            pred = pred[:-4]
        if not pred.strip():
            score = 0.0
        elif similarity:
            score = f1_score(pred, ref)
        elif "_mf" in dataname:
            score = exact_match_score(pred, ref, normalize=False)
        elif "summary" in dataname:
            score = rouge_score(pred, ref)
        elif "qa_eng" in dataname:
            score = max(f1_score(pred, ref), include_score(pred, ref))
        elif "choice_eng" in dataname:
            score = include_score(pred.split("\n")[0], ref)
        elif dataname == "gsm":
            pred = pred.strip().lower().split("the answer is ")[-1]
            score = include_score_gsm(pred, ref, normalize=False)
        else:
            score = include_score(pred, ref)
        scores.append(float(score))
    return scores
