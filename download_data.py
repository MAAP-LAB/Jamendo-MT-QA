#!/usr/bin/env python3
"""Download Jamendo-MT-QA benchmark data from HuggingFace.

Downloads:
  - Audio (WAV bytes) from `m-a-a-p/Jamendo-QA` -> `data/audio/<track_id>.wav`
  - Single-track QA  (`cap`)  from `m-a-a-p/Jamendo-QA`  qa_v2.json
  - Multi-track  QA  (`multi`) from `m-a-a-p/Jamendo-MT-QA`

Usage:
    python download_data.py --mode all       # download everything
    python download_data.py --mode cap       # single-track QA + audio only
    python download_data.py --mode multi     # multi-track QA + audio only
    python download_data.py --mode questions # JSON questions only (no audio)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm


JAMENDO_QA_REPO = "m-a-a-p/Jamendo-QA"
JAMENDO_MTQA_REPO = "m-a-a-p/Jamendo-MT-QA"


def download_questions(out_dir: Path, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("cap", "all", "questions"):
        print(f"[questions] Downloading single-track QA (qa_v2.json) from {JAMENDO_QA_REPO}")
        path = hf_hub_download(
            repo_id=JAMENDO_QA_REPO,
            filename="qa_v2.json",
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        print(f"[questions]   -> {path}")

    if mode in ("multi", "all", "questions"):
        print(f"[questions] Downloading multi-track QA from {JAMENDO_MTQA_REPO}")
        ds = load_dataset(JAMENDO_MTQA_REPO, split="train")
        out_path = out_dir / "multi_qa.json"
        records = [dict(row) for row in ds]
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"[questions]   -> {out_path}  ({len(records)} QA pairs)")


def download_audio(out_dir: Path, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[audio] Streaming {JAMENDO_QA_REPO} (gated; accept the dataset terms on HF first)")
    ds = load_dataset(JAMENDO_QA_REPO, split="train", streaming=True)

    n_written = 0
    for row in tqdm(ds, desc="audio"):
        audio_path = row.get("audio_path") or ""
        track_id = Path(audio_path).stem or row.get("track_id") or f"track_{n_written}"
        wav_bytes = row.get("audio_bytes")
        if wav_bytes is None:
            continue
        target = out_dir / f"{track_id}.wav"
        if target.exists():
            continue
        target.write_bytes(wav_bytes)
        n_written += 1
        if limit is not None and n_written >= limit:
            break

    print(f"[audio] wrote {n_written} files to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode", choices=["cap", "multi", "all", "questions"], default="all",
                        help="cap = single-track only; multi = multi-track only; all = both; "
                             "questions = JSON files only (skip audio)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root directory for downloads (default: ./data)")
    parser.add_argument("--limit", type=int, default=None,
                        help="(debug) Stop after N audio files")
    args = parser.parse_args()

    questions_dir = args.data_dir / "questions"
    audio_dir = args.data_dir / "audio"

    download_questions(questions_dir, args.mode)
    if args.mode != "questions":
        download_audio(audio_dir, limit=args.limit)

    print("\nDone. Layout:")
    print(f"  {questions_dir}/qa_v2.json       (cap / single-track QA)")
    print(f"  {questions_dir}/multi_qa.json    (multi-track QA)")
    print(f"  {audio_dir}/<track_id>.wav      (audio)")


if __name__ == "__main__":
    main()
