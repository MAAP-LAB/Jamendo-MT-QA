#!/usr/bin/env python3
"""Run a custom QA model against the Jamendo-MT-QA benchmark.

Two modes (matching the paper):

  --mode multi  Your adapter implements `predict_multi(a1, a2, q, type)` and
                consumes both audio files end-to-end.

  --mode cap    Your adapter implements `caption(audio_path)`. The runner
                captions every track, then asks a *comparator* text LLM
                (OpenAI / Anthropic) to answer each comparative question
                from the two captions + the question.

Outputs a JSON list of `{pair_id, audio1, audio2, qa_type, question,
gold_answer, prediction}` records suitable for `evaluate.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

from adapters import BaseAdapter, load_adapter


CAP_COMPARATOR_PROMPT = """You are a music comparison expert. Two music tracks are described below by their captions. Answer the comparative question concisely.

## Track 1 ({audio1}) caption
{caption1}

## Track 2 ({audio2}) caption
{caption2}

## Question type
{qa_type}  ({qa_type_hint})

## Question
{question}

Output only the answer (no explanation, no JSON)."""


QA_TYPE_HINTS = {
    "yes_no": "respond with exactly 'yes' or 'no'",
    "short_answer": "respond with the track identifier (e.g. 'track1' or the audio name)",
    "sentence": "respond with one comparative sentence",
}


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of QA records in {path}, got {type(data)}")
    return data


def normalize_records(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten both supported question schemas into a uniform list of items."""
    out: List[Dict[str, Any]] = []
    for entry in raw:
        if "qa_pairs" in entry:
            audio1 = entry.get("audio1") or entry.get("track1")
            audio2 = entry.get("audio2") or entry.get("track2")
            pair_id = entry.get("pair_id") or f"{audio1}__{audio2}"
            for qa in entry["qa_pairs"]:
                out.append({
                    "pair_id": pair_id,
                    "audio1": audio1,
                    "audio2": audio2,
                    "qa_type": qa.get("type") or qa.get("qa_type"),
                    "question": qa["question"],
                    "gold_answer": qa.get("answer"),
                })
        else:
            out.append({
                "pair_id": entry.get("pair_id"),
                "audio1": entry.get("audio1") or entry.get("track1"),
                "audio2": entry.get("audio2") or entry.get("track2"),
                "qa_type": entry.get("qa_type") or entry.get("type"),
                "question": entry["question"],
                "gold_answer": entry.get("answer"),
            })
    return out


def resolve_audio(audio_dir: Path, name: str) -> str:
    for ext in (".wav", ".mp3", ".flac", ".ogg"):
        p = audio_dir / f"{name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Audio not found for '{name}' under {audio_dir} (tried .wav/.mp3/.flac/.ogg)")


# --------------------------------------------------------------------------- #
# Comparator (Cap mode)                                                        #
# --------------------------------------------------------------------------- #

def make_comparator(provider: str, model: str):
    """Return a callable (caption1, caption2, question, qa_type, audio1, audio2) -> str."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install `openai` to use the OpenAI comparator.") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for --comparator-provider openai.")
        client = OpenAI(api_key=api_key)

        def call(caption1, caption2, question, qa_type, audio1, audio2):
            prompt = CAP_COMPARATOR_PROMPT.format(
                caption1=caption1, caption2=caption2,
                question=question, qa_type=qa_type,
                qa_type_hint=QA_TYPE_HINTS.get(qa_type, ""),
                audio1=audio1, audio2=audio2,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()

        return call

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Install `anthropic` to use the Anthropic comparator.") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for --comparator-provider anthropic.")
        client = anthropic.Anthropic(api_key=api_key)

        def call(caption1, caption2, question, qa_type, audio1, audio2):
            prompt = CAP_COMPARATOR_PROMPT.format(
                caption1=caption1, caption2=caption2,
                question=question, qa_type=qa_type,
                qa_type_hint=QA_TYPE_HINTS.get(qa_type, ""),
                audio1=audio1, audio2=audio2,
            )
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()

        return call

    raise ValueError(f"Unknown comparator provider: {provider}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run_multi(adapter: BaseAdapter, items: Iterable[Dict], audio_dir: Path) -> List[Dict]:
    results = []
    for item in tqdm(list(items), desc="multi"):
        a1 = resolve_audio(audio_dir, item["audio1"])
        a2 = resolve_audio(audio_dir, item["audio2"])
        try:
            pred = adapter.predict_multi(a1, a2, item["question"], item["qa_type"])
        except Exception as e:
            pred = f"<ERROR: {e}>"
        results.append({**item, "prediction": pred})
    return results


def run_cap(
    adapter: BaseAdapter,
    items: List[Dict],
    audio_dir: Path,
    comparator,
    cache_path: Optional[Path] = None,
) -> List[Dict]:
    captions: Dict[str, str] = {}
    if cache_path and cache_path.exists():
        captions = json.loads(cache_path.read_text())

    unique_tracks = sorted({i["audio1"] for i in items} | {i["audio2"] for i in items})
    for track in tqdm(unique_tracks, desc="cap-captions"):
        if track in captions:
            continue
        try:
            captions[track] = adapter.caption(resolve_audio(audio_dir, track))
        except Exception as e:
            captions[track] = f"<ERROR: {e}>"
        if cache_path:
            cache_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2))

    results = []
    for item in tqdm(items, desc="cap-compare"):
        c1 = captions[item["audio1"]]
        c2 = captions[item["audio2"]]
        try:
            pred = comparator(c1, c2, item["question"], item["qa_type"], item["audio1"], item["audio2"])
        except Exception as e:
            pred = f"<ERROR: {e}>"
        results.append({**item, "prediction": pred, "caption1": c1, "caption2": c2})
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", choices=["cap", "multi"], required=True)
    p.add_argument("--adapter", required=True,
                   help="`module.path:ClassName` (e.g. adapters.qwen2_audio:Qwen2AudioAdapter)")
    p.add_argument("--questions", type=Path, default=Path("data/questions/multi_qa.json"),
                   help="Path to multi-track QA JSON (from download_data.py).")
    p.add_argument("--audio-dir", type=Path, default=Path("data/audio"),
                   help="Directory containing <track_id>.wav files.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write predictions JSON.")
    p.add_argument("--limit", type=int, default=None,
                   help="(debug) Run only the first N items.")
    p.add_argument("--caption-cache", type=Path, default=None,
                   help="(cap mode) JSON file for caching per-track captions.")
    p.add_argument("--comparator-provider", choices=["openai", "anthropic"], default="openai",
                   help="(cap mode) Text LLM used to compare captions.")
    p.add_argument("--comparator-model", default="gpt-4o-mini",
                   help="(cap mode) Comparator model name.")
    args = p.parse_args()

    if not args.questions.exists():
        sys.exit(f"Questions file not found: {args.questions}\nRun `python download_data.py` first.")
    if not args.audio_dir.exists():
        sys.exit(f"Audio directory not found: {args.audio_dir}\nRun `python download_data.py` first.")

    raw = load_questions(args.questions)
    items = normalize_records(raw)
    if args.limit:
        items = items[: args.limit]
    print(f"Loaded {len(items)} QA items.")

    adapter = load_adapter(args.adapter)
    print(f"Loaded adapter: {args.adapter}")

    if args.mode == "multi":
        results = run_multi(adapter, items, args.audio_dir)
    else:
        comparator = make_comparator(args.comparator_provider, args.comparator_model)
        results = run_cap(adapter, items, args.audio_dir, comparator, args.caption_cache)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
