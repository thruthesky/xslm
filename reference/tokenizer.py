import json
from collections import Counter
from pathlib import Path


def merge(ids, pair, new_id):
    result, i = [], 0
    while i < len(ids):
        if tuple(ids[i:i + 2]) == tuple(pair):
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class BPE:
    def __init__(self, vocab, merges):
        self.vocab, self.merges = vocab, merges
        self.chars = {c: i for i, c in enumerate(vocab) if len(c) == 1}

    @classmethod
    def train(cls, text, count):
        vocab = ['\ufffd'] + sorted(set(text) - {'\ufffd'})
        ids = [vocab.index(c) for c in text]
        merges = []
        for _ in range(count):
            pairs = Counter((a, b) for a, b in zip(ids, ids[1:])
                            if '\n' not in vocab[a] + vocab[b])
            if not pairs:
                break
            pair, frequency = pairs.most_common(1)[0]
            if frequency < 2:
                break
            new_id = len(vocab)
            vocab.append(vocab[pair[0]] + vocab[pair[1]])
            merges.append([list(pair), new_id])
            ids = merge(ids, pair, new_id)
        return cls(vocab, merges)

    def encode(self, text):
        ids = [self.chars.get(c, 0) for c in text]
        for pair, new_id in self.merges:
            ids = merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)

    def state(self):
        return {'vocab': self.vocab, 'merges': self.merges}


if __name__ == '__main__':
    settings = json.loads(Path('settings.json').read_text())
    text = Path('data/train.txt').read_text(encoding='utf-8')
    tokenizer = BPE.train(text, settings['merges'])
    Path('data/tokenizer.json').write_text(
        json.dumps(tokenizer.state(), ensure_ascii=False), encoding='utf-8')
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print('vocab:', len(tokenizer.vocab), 'merges:', len(tokenizer.merges))
    print('roundtrip: OK', tokenizer.encode('Q: hi\nA: '))
