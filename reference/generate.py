import argparse
import json
import math
import random
from pathlib import Path
from model import GPT
from tokenizer import BPE


def load():
    saved = json.loads(Path('checkpoints/model.json').read_text())
    cfg = saved['settings']
    tokenizer = BPE(**saved['tokenizer'])
    model = GPT(len(tokenizer.vocab), cfg['width'], cfg['context'], cfg['seed'])
    model.load(saved['weights'])
    return model, tokenizer, cfg


def generate(model, tokenizer, prompt, tokens=24, temperature=0.8, top_k=8, seed=7):
    if not prompt or tokens < 1 or temperature < 0 or not math.isfinite(temperature) or top_k < 0:
        raise ValueError('invalid sampling parameters')
    ids, result, rng = tokenizer.encode(prompt), [], random.Random(seed)
    for _ in range(tokens):
        logits = model.forward(ids[-model.context:])[-1]
        ranked = sorted(range(len(logits)), key=lambda i: logits[i].data, reverse=True)
        if temperature == 0:
            token = ranked[0]
        else:
            candidates = ranked[:top_k] if top_k else ranked
            peak = logits[candidates[0]].data
            weights = [math.exp((logits[i].data - peak) / temperature) for i in candidates]
            token = rng.choices(candidates, weights=weights)[0]
        ids.append(token)
        if tokenizer.vocab[token] == '\n':
            break
        result.append(token)
    return tokenizer.decode(result)


if __name__ == '__main__':
    model, tokenizer, cfg = load()
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='Q: hi\nA: ')
    parser.add_argument('--tokens', type=int, default=cfg['tokens'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--seed', type=int, default=cfg['seed'])
    args = parser.parse_args()
    assert 1 <= args.tokens <= cfg['max_tokens']
    print(generate(model, tokenizer, args.prompt, args.tokens,
                   args.temperature, args.top_k, args.seed))
