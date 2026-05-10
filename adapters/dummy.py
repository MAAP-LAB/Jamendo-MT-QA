"""Trivial adapter — returns canned answers. Useful for sanity-checking the pipeline."""

from .base import BaseAdapter


class DummyAdapter(BaseAdapter):
    def caption(self, audio_path: str) -> str:
        return f"A music track at {audio_path} (DummyAdapter caption — replace with a real model)."

    def predict_multi(self, audio1_path, audio2_path, question, qa_type):
        if qa_type == "yes_no":
            return "no"
        if qa_type == "short_answer":
            return "track1"
        return "DummyAdapter cannot meaningfully compare the two tracks."
