"""Adapter interface for plugging custom QA models into the benchmark.

The benchmark has two paradigms (matching the paper, Section 4.1):

  * **Multi**  — your model directly consumes (audio1, audio2, question)
                and produces an answer end-to-end.
                Override `predict_multi(...)`.

  * **Cap**    — your model only consumes a single audio. Each track is first
                converted to a natural-language caption by your model, and a
                separate text LLM (the "comparator") performs comparative
                reasoning over the two captions + the question.
                Override `caption(audio_path)`. The comparator is configured
                via `--comparator-*` flags on `run_benchmark.py`.

`qa_type` is one of {"yes_no", "short_answer", "sentence"}. Use it if you
want to format prompts differently per question type.
"""

from __future__ import annotations

import importlib
from typing import Type


class BaseAdapter:
    """Subclass and override `caption` (for Cap mode) and/or `predict_multi`."""

    def caption(self, audio_path: str) -> str:
        """Return a natural-language description of a single audio file.

        Used by Cap mode only. The runner caches captions per track so this is
        called at most once per unique track.
        """
        raise NotImplementedError("This adapter does not support `cap` mode.")

    def predict_multi(
        self,
        audio1_path: str,
        audio2_path: str,
        question: str,
        qa_type: str,
    ) -> str:
        """Return an answer string for a comparative question over two tracks.

        Used by Multi mode only.
        """
        raise NotImplementedError("This adapter does not support `multi` mode.")


def load_adapter(spec: str) -> BaseAdapter:
    """Load `module.path:ClassName` and return an instance."""
    if ":" not in spec:
        raise ValueError(f"Adapter spec must be 'module.path:ClassName', got: {spec}")
    module_path, class_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    cls: Type[BaseAdapter] = getattr(module, class_name)
    instance = cls()
    if not isinstance(instance, BaseAdapter):
        raise TypeError(f"{spec} must subclass adapters.BaseAdapter")
    return instance
