#!/usr/bin/env python3
"""Evaluate predictions against Jamendo-MT-QA gold answers.

Reports:
  - yes/no  -> accuracy
  - short_answer -> exact-match accuracy
  - sentence -> BLEU, ROUGE-1/2/L, BERTScore (optional, see --bertscore)
  - LLM-as-a-Judge sentence score (1-5) using OpenAI or Anthropic
    (matches the protocol described in the paper, Section 4.2 / Appendix D).

Inputs:
  predictions JSON produced by `run_benchmark.py` — list of records with
  fields {audio1, audio2, qa_type, question, gold_answer, prediction}.

API keys are taken from the environment:
  OPENAI_API_KEY / ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Surface-form metrics                                                         #
# --------------------------------------------------------------------------- #

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def exact_match(pred: str, gold: str) -> bool:
    p = normalize(pred)
    g = normalize(gold)
    if not p or not g:
        return False
    return g in p or p in g


def yes_no_match(pred: str, gold: str) -> bool:
    p = normalize(pred)
    g = normalize(gold)
    pred_label = "yes" if p.startswith("yes") or p == "true" else ("no" if p.startswith("no") or p == "false" else None)
    gold_label = "yes" if g.startswith("yes") else ("no" if g.startswith("no") else None)
    return pred_label is not None and pred_label == gold_label


def compute_text_metrics(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """BLEU + ROUGE-1/2/L on parallel lists. Returns 0s if metrics aren't installed."""
    out: Dict[str, float] = {}
    try:
        from sacrebleu import corpus_bleu
        out["bleu"] = corpus_bleu(hyps, [refs]).score
    except ImportError:
        out["bleu"] = float("nan")
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for r, h in zip(refs, hyps):
            s = scorer.score(r or "", h or "")
            for k in sums:
                sums[k] += s[k].fmeasure
        n = max(1, len(refs))
        out["rouge1"] = 100 * sums["rouge1"] / n
        out["rouge2"] = 100 * sums["rouge2"] / n
        out["rougeL"] = 100 * sums["rougeL"] / n
    except ImportError:
        out["rouge1"] = out["rouge2"] = out["rougeL"] = float("nan")
    return out


def compute_bertscore(refs: List[str], hyps: List[str], lang: str = "en") -> Optional[float]:
    try:
        from bert_score import score
    except ImportError:
        return None
    P, R, F1 = score(hyps, refs, lang=lang, rescale_with_baseline=False, verbose=False)
    return float(F1.mean())


# --------------------------------------------------------------------------- #
# LLM-as-a-Judge (sentence-level)                                             #
# --------------------------------------------------------------------------- #

JUDGE_PROMPT = """You are an expert evaluator for multi-track music question answering.

You will see two music tracks (referenced by name only — captions are not available),
a comparative question, the gold answer, and the model's predicted answer.

Rate the prediction on a 1-5 scale for each criterion:

1. Correctness: Does the prediction agree with the gold answer in substance?
2. Comparative Validity: Does the prediction make a meaningful comparison between the two tracks?
3. Reasoning Quality: Is the prediction coherent and well-formed?

Return ONLY a JSON object: {{"correctness": <1-5>, "comparative_validity": <1-5>, "reasoning_quality": <1-5>}}

## Track 1: {audio1}
## Track 2: {audio2}
## Question: {question}
## Gold Answer: {gold}
## Prediction: {pred}
"""


def make_judge(provider: str, model: str):
    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --judge-provider openai.")
        client = OpenAI(api_key=api_key)

        def call(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        return call

    if provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for --judge-provider anthropic.")
        client = anthropic.Anthropic(api_key=api_key)

        def call(prompt: str) -> str:
            resp = client.messages.create(
                model=model, max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        return call

    raise ValueError(f"Unknown judge provider: {provider}")


def parse_judge_json(raw: str) -> Optional[Dict[str, int]]:
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return {k: int(obj[k]) for k in ("correctness", "comparative_validity", "reasoning_quality") if k in obj}
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Path to write the metrics JSON.")
    p.add_argument("--bertscore", action="store_true", help="Compute BERTScore (requires `bert_score`).")
    p.add_argument("--judge", action="store_true", help="Run LLM-as-a-Judge on sentence-level items.")
    p.add_argument("--judge-provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-limit", type=int, default=None,
                   help="(debug) Cap the number of judged items.")
    args = p.parse_args()

    records: List[Dict] = json.loads(args.predictions.read_text())

    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_type[r.get("qa_type", "unknown")].append(r)

    metrics: Dict[str, Any] = {"counts": {k: len(v) for k, v in by_type.items()}}

    # Yes/No accuracy
    yn = by_type.get("yes_no", [])
    if yn:
        correct = sum(yes_no_match(r["prediction"], r["gold_answer"]) for r in yn if r.get("gold_answer"))
        metrics["yes_no_accuracy"] = correct / max(1, len(yn))

    # Short-answer accuracy (exact match, case-insensitive substring)
    sa = by_type.get("short_answer", [])
    if sa:
        correct = sum(exact_match(r["prediction"], r["gold_answer"]) for r in sa if r.get("gold_answer"))
        metrics["short_answer_accuracy"] = correct / max(1, len(sa))

    # Sentence-level surface metrics
    sent = by_type.get("sentence", [])
    if sent:
        refs = [r.get("gold_answer") or "" for r in sent]
        hyps = [r.get("prediction") or "" for r in sent]
        metrics["sentence"] = compute_text_metrics(refs, hyps)
        if args.bertscore:
            bs = compute_bertscore(refs, hyps)
            if bs is not None:
                metrics["sentence"]["bertscore_f1"] = bs

    # LLM-as-a-Judge (sentence-level)
    if args.judge and sent:
        judge = make_judge(args.judge_provider, args.judge_model)
        target = sent if args.judge_limit is None else sent[: args.judge_limit]
        scores = {"correctness": [], "comparative_validity": [], "reasoning_quality": []}
        for r in tqdm(target, desc="judge"):
            prompt = JUDGE_PROMPT.format(
                audio1=r["audio1"], audio2=r["audio2"],
                question=r["question"], gold=r.get("gold_answer", ""),
                pred=r["prediction"],
            )
            try:
                parsed = parse_judge_json(judge(prompt))
            except Exception as e:
                parsed = None
                print(f"  judge error: {e}", file=sys.stderr)
            if parsed:
                for k, v in parsed.items():
                    scores[k].append(v)
        metrics["judge"] = {
            f"{k}_mean": (sum(v) / len(v)) if v else None for k, v in scores.items()
        }
        metrics["judge"]["n_scored"] = len(scores["correctness"])
        metrics["judge"]["model"] = f"{args.judge_provider}:{args.judge_model}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
