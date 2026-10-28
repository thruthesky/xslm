# XSLM — Extra Small Language Model

**XSLM (Extra Small Language Model)** is a language model with **millions to a few hundred million parameters** — one step smaller than LLMs and SLMs. It is built to run on small devices and to do one job well.

The term **XSLM** was coined by [SONUB (sonub.com)](https://sonub.com) in October 2026, in its course for learning how to build an AI from scratch. This repository is the public definition of the term and a minimal reference implementation.

- Website: https://thruthesky.github.io/xslm/
- Build one step by step in your web browser: https://sonub.com/ai/build

## Size: LLM, SLM, XSLM

| Model | Parameters | Where it runs |
| --- | --- | --- |
| LLM — Large Language Model | tens to hundreds of billions | GPU servers in data centers |
| SLM — Small Language Model | around a few billion | a powerful computer or a single GPU |
| **XSLM — Extra Small Language Model** | **millions to a few hundred million** | smartphones, wearables, IoT devices |

## Three features

1. **On-device** — light enough to run on its own, without a network, on smartphones, wearables and IoT equipment.
2. **Very low cost** — an SLM needs several GB of VRAM just to load; an XSLM runs in a few hundred MB of memory or less.
3. **A specialist for one job** — not an encyclopedia like an LLM, but one task done well: summarizing, classifying text, function calling, or routing for an AI agent.

Public models of this size already exist, for example SmolLM2-135M (Hugging Face, 2024, about 135 million parameters) and MobileBERT (Google, 2020, about 25 million parameters). Two common ways to shrink a big model down to this size are **knowledge distillation** (a small student model learns the probabilities of a big teacher model) and **quantization** (8-bit or 4-bit numbers instead of 32-bit).

## How to build an XSLM — the basic method

An XSLM is a chain of small programs. Each program reads the file made by the previous one and makes the next file.

| Step | File | Reads → Makes | What it does |
| --- | --- | --- | --- |
| 1. Data | `prepare.py` | `data/raw.txt` → `data/train.txt`, `data/val.txt` | Cleans Q/A lines and splits them into training and validation text |
| 2. Tokenizer | `tokenizer.py` | `train.txt` → `data/tokenizer.json` | Character-level BPE: learns merge rules, turns text into token IDs and back |
| 3. Model | `engine.py`, `model.py` | — | Autograd (`Value`, `backward`) and a GPT: token + position embedding, causal attention, MLP, residuals, output head |
| 4. Training | `train.py` | data + tokenizer → `checkpoints/model.json` | Next-token prediction, cross-entropy loss, gradient descent, resumable checkpoints |
| 5. Generation | `generate.py` | checkpoint → text | Repeats "predict the next token, pick one" with temperature, top-k and seed |
| 6. Runner | `runner.py` | checkpoint → HTTP API | `GET /health`, `POST /generate` on `127.0.0.1:8765` |
| 7. XChat | `xchat.py` | your question → Runner | A terminal chat client: XChat → Runner → XSLM |
| 8. Publish | `export.py` | checkpoint → `hub/` | Writes `model.safetensors`, `config.json`, `tokenizer.json` for Hugging Face |

When an earlier file changes, rebuild everything after it (for example: edit `raw.txt` → run prepare, tokenizer and train again).

## Run the reference model

The [`reference/`](reference/) folder is the complete course code: pure Python 3, standard library only — no `pip`, no GPU, no API key, no pretrained weights.

```bash
git clone https://github.com/thruthesky/xslm
cd xslm/reference
python3 prepare.py        # train 12, val 3
python3 tokenizer.py      # vocab: 40 merges: 12 / roundtrip: OK
python3 engine.py         # y = 15.0 dy/dx = 8.0
python3 train.py          # parameters: 1248 ... saved: checkpoints/model.json
python3 generate.py       # continues "Q: hi\nA: "
python3 runner.py &       # XSLM Runner: ('127.0.0.1', 8765)
python3 xchat.py          # You> hi
```

The reference model is a learning model with **1,248 parameters** (vocabulary 40, width 8, context 12, 1 block, 1 head; settings in `reference/settings.json`). After 20 training steps its answers are still mostly random — that is expected. It is shrunk to the smallest size so you can read every calculation; the parts are the same as in a real XSLM. Grow the same structure and train it on data for one job, and it becomes a practical XSLM.

## Learn it step by step

The full course — 10 steps, each with explanations, experiments and an in-browser Linux terminal — is at **https://sonub.com/ai/build**. Everything is typed and trained inside the web browser; nothing to install.

---

<sub>Author: JaeHo, Song · Contact: [thruthesky@gmail.com](mailto:thruthesky@gmail.com)</sub>
