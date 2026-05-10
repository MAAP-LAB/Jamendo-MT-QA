# Jamendo-MT-QA

A benchmark for **multi-track comparative music question answering**.
36,519 QA items over 12,173 Jamendo track pairs, with three question types
(yes/no, short-answer, sentence-level). Built on top of
[Jamendo-QA](https://huggingface.co/datasets/m-a-a-p/Jamendo-QA).

- **Project page:** https://maap-lab.github.io/Jamendo-MT-QA/
- **Dataset (HF):** https://huggingface.co/datasets/m-a-a-p/Jamendo-MT-QA
- **Audio (HF):** https://huggingface.co/datasets/m-a-a-p/Jamendo-QA *(gated — accept the terms first)*

This repository provides the **runner** for evaluating your own QA model
against Jamendo-MT-QA. It does not include the dataset itself; the
download script pulls everything from HuggingFace.

---

## 1. Install

```bash
git clone https://github.com/MAAP-LAB/Jamendo-MT-QA.git
cd Jamendo-MT-QA
pip install -r requirements.txt
```

You will also need to install whatever runtime your model needs
(e.g. `pip install transformers torch torchaudio librosa` for the
provided Qwen2-Audio adapter).

## 2. Download the data

You must have a HuggingFace account and have **accepted the dataset terms**
on the [Jamendo-QA dataset page](https://huggingface.co/datasets/m-a-a-p/Jamendo-QA).
Then authenticate locally:

```bash
huggingface-cli login
```

```bash
# Audio (Jamendo-QA) + multi-track questions (Jamendo-MT-QA)
python download_data.py --mode all
```

Layout after download:

```
data/
  audio/<track_id>.wav
  questions/multi_qa.json   # comparative QA over track pairs
  questions/qa_v2.json      # single-track QA from Jamendo-QA (optional)
```

## 3. Run the benchmark

The benchmark has two modes, matching the paper (Section 4.1):

| Mode    | When to use                                                      | Adapter method      |
| ------- | ---------------------------------------------------------------- | ------------------- |
| `multi` | Your model takes **two audio clips** and answers end-to-end.     | `predict_multi(...)` |
| `cap`   | Your model handles **one audio at a time**. Each track is captioned separately, and a text LLM compares the two captions to answer. | `caption(audio_path)` |

### Implement an adapter

Subclass `BaseAdapter` and override the relevant method:

```python
# my_adapter.py
from adapters import BaseAdapter

class MyModel(BaseAdapter):
    def __init__(self):
        # load your model here
        ...

    # For multi mode
    def predict_multi(self, audio1_path, audio2_path, question, qa_type):
        ...
        return "answer string"

    # For cap mode
    def caption(self, audio_path):
        ...
        return "single-track caption"
```

Then point the runner at it:

```bash
# Multi mode
python run_benchmark.py \
    --mode multi \
    --adapter my_adapter:MyModel \
    --output predictions/my_model_multi.json

# Cap mode (requires an OpenAI or Anthropic key for the comparator)
export OPENAI_API_KEY=...
python run_benchmark.py \
    --mode cap \
    --adapter my_adapter:MyModel \
    --comparator-provider openai \
    --comparator-model gpt-4o-mini \
    --caption-cache predictions/my_model_captions.json \
    --output predictions/my_model_cap.json
```

We ship two adapters under `adapters/`:

- `adapters.dummy:DummyAdapter` — canned outputs, for sanity-checking the
  pipeline.
- `adapters.qwen2_audio:Qwen2AudioAdapter` — reference adapter for
  `Qwen/Qwen2-Audio-7B-Instruct`. Implements both `caption()` and
  `predict_multi()`.

### Run only on a subset

```bash
python run_benchmark.py --mode multi --adapter ... --output ... --limit 100
```

## 4. Evaluate

`evaluate.py` reads the predictions JSON and reports:

- yes/no — accuracy
- short-answer — exact-match accuracy
- sentence-level — BLEU, ROUGE-1/2/L, optional BERTScore
- (optional) LLM-as-a-Judge sentence score (1–5), as in the paper.

```bash
# Surface-form metrics only
python evaluate.py \
    --predictions predictions/my_model_multi.json \
    --output eval_results/my_model_multi.json

# Add BERTScore + LLM judge
export OPENAI_API_KEY=...
python evaluate.py \
    --predictions predictions/my_model_multi.json \
    --output eval_results/my_model_multi.json \
    --bertscore \
    --judge --judge-provider openai --judge-model gpt-4o-mini
```

## API keys

This repo never ships any API key. The runner and evaluator read
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` **from your environment only**,
and the included `.gitignore` excludes `.env*`. Do not commit secrets.

## Citation

If you use Jamendo-MT-QA, please cite:

```bibtex
@misc{koh2026jamendomtqabenchmarkmultitrackcomparative,
      title={Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering},
      author={Junyoung Koh and Jaeyun Lee and Soo Yong Kim and Gyu Hyeong Choi and Jung In Koh and Jordan Phillips and Yeonjin Lee and Min Song},
      year={2026},
      eprint={2604.09721},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2604.09721},
}
```

## License

Code: MIT. Audio is Creative Commons-licensed via the Jamendo platform —
see the [Jamendo-QA dataset card](https://huggingface.co/datasets/m-a-a-p/Jamendo-QA)
for full terms. The benchmark is intended for research use only.
