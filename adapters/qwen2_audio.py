"""Reference adapter for Qwen/Qwen2-Audio-7B-Instruct.

Supports both Cap mode (single-audio captioning) and Multi mode
(joint two-audio reasoning).

Install the extra deps before using:
    pip install transformers torchaudio librosa

Usage:
    # Cap mode
    python run_benchmark.py --mode cap   --adapter adapters.qwen2_audio:Qwen2AudioAdapter ...
    # Multi mode
    python run_benchmark.py --mode multi --adapter adapters.qwen2_audio:Qwen2AudioAdapter ...
"""

from __future__ import annotations

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from .base import BaseAdapter


SAMPLING_RATE = 16000
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


class Qwen2AudioAdapter(BaseAdapter):
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=self.device
        )
        self.model.eval()

    def _load(self, path: str):
        audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
        return audio

    def _generate(self, conversation, audios, max_new_tokens: int = 256) -> str:
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=SAMPLING_RATE
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out = out[:, inputs["input_ids"].size(1):]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def caption(self, audio_path):
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": (
                    "Describe this music track in detail, covering genre, tempo, key, "
                    "instrumentation, mood, vocal characteristics, and production style."
                )},
            ]},
        ]
        return self._generate(conversation, [self._load(audio_path)], max_new_tokens=400)

    def predict_multi(self, audio1_path, audio2_path, question, qa_type):
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio1_path},
                {"type": "audio", "audio_url": audio2_path},
                {"type": "text", "text": (
                    "You are given two music tracks (Track 1 first, Track 2 second). "
                    f"Question: {question}\n"
                    "Answer concisely; reference both tracks where appropriate."
                )},
            ]},
        ]
        return self._generate(conversation, [self._load(audio1_path), self._load(audio2_path)])
