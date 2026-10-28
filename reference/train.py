import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from model import GPT
from tokenizer import BPE


def main():
    cfg = json.loads(Path('settings.json').read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=cfg['steps'])
    args = parser.parse_args()
    assert args.steps > 0, '--steps must be positive'
    token_state = json.loads(Path('data/tokenizer.json').read_text())
    tokenizer = BPE(**token_state)
    train_text = Path('data/train.txt').read_text(encoding='utf-8')
    val_text = Path('data/val.txt').read_text(encoding='utf-8')
    fingerprint = hashlib.sha256((train_text + '\0' + val_text).encode()).hexdigest()
    train, val = tokenizer.encode(train_text), tokenizer.encode(val_text)
    model = GPT(len(tokenizer.vocab), cfg['width'], cfg['context'], cfg['seed'])
    rng, step = random.Random(cfg['seed']), 0
    checkpoint = Path('checkpoints/model.json')
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        assert saved['settings'] == cfg and saved['tokenizer'] == token_state, 'changed settings/tokenizer: start a new checkpoint'
        assert saved['data_hash'] == fingerprint, 'changed data: start a new checkpoint'
        model.load(saved['weights'])
        step = saved['step']
        rng.setstate((saved['random'][0], tuple(saved['random'][1]), saved['random'][2]))
    n = model.context
    assert len(train) > n and len(val) > n, 'add more text'
    print('parameters:', len(model.parameters), 'resume:', step, flush=True)
    print('validation:', round(model.loss(val[:n], val[1:n + 1]).data, 4), flush=True)
    checkpoint.parent.mkdir(exist_ok=True)
    for _ in range(args.steps):
        start = rng.randrange(len(train) - n)
        window = train[start:start + n + 1]
        for p in model.parameters:
            p.grad = 0
        loss = model.loss(window[:-1], window[1:])
        loss.backward()
        size = math.sqrt(sum(p.grad * p.grad for p in model.parameters))
        scale = min(1.0, 1.0 / (size + 1e-8))
        for p in model.parameters:
            p.data -= cfg['learning_rate'] * p.grad * scale
        step += 1
        state = {'settings': cfg, 'tokenizer': token_state, 'data_hash': fingerprint,
                 'weights': model.state(), 'step': step, 'random': rng.getstate()}
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, checkpoint)
        print('step:', step, 'loss:', round(loss.data, 4), flush=True)
    print('validation:', round(model.loss(val[:n], val[1:n + 1]).data, 4), flush=True)
    print('saved:', checkpoint, flush=True)


if __name__ == '__main__':
    main()
