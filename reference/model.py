import random
from engine import Value, linear, norm, softmax


class GPT:
    def __init__(self, vocab_size, width, context, seed=7):
        self.width, self.context = width, context
        rng = random.Random(seed)
        shapes = {
            'token': (vocab_size, width), 'position': (context, width),
            'query': (width, width), 'key': (width, width),
            'value': (width, width), 'project': (width, width),
            'expand': (2 * width, width), 'contract': (width, 2 * width),
            'head': (vocab_size, width),
        }
        self.weights = {
            name: [[Value(rng.gauss(0, 0.1)) for _ in range(cols)]
                   for _ in range(rows)] for name, (rows, cols) in shapes.items()
        }
        self.parameters = [v for matrix in self.weights.values()
                           for row in matrix for v in row]

    def forward(self, ids):
        assert 0 < len(ids) <= self.context
        w, keys, values, output = self.weights, [], [], []
        for position, token in enumerate(ids):
            x = [a + b for a, b in zip(w['token'][token], w['position'][position])]
            h = norm(x)
            q = linear(h, w['query'])
            keys.append(linear(h, w['key']))
            values.append(linear(h, w['value']))
            scores = [sum(a * b for a, b in zip(q, k)) / self.width ** 0.5
                      for k in keys]
            attention = softmax(scores)
            mixed = [sum(p * v[i] for p, v in zip(attention, values))
                     for i in range(self.width)]
            x = [a + b for a, b in zip(x, linear(mixed, w['project']))]
            hidden = [v.relu() for v in linear(norm(x), w['expand'])]
            x = [a + b for a, b in zip(x, linear(hidden, w['contract']))]
            output.append(linear(norm(x), w['head']))
        return output

    def loss(self, inputs, targets):
        logits = self.forward(inputs)
        losses = [-softmax(row)[target].log()
                  for row, target in zip(logits, targets)]
        return sum(losses) / len(losses)

    def state(self):
        return {name: [[v.data for v in row] for row in matrix]
                for name, matrix in self.weights.items()}

    def load(self, state):
        assert set(state) == set(self.weights), 'checkpoint keys'
        for name, matrix in self.weights.items():
            assert len(state[name]) == len(matrix), 'checkpoint rows'
            for row, saved in zip(matrix, state[name]):
                assert len(saved) == len(row), 'checkpoint columns'
                for value, number in zip(row, saved):
                    value.data = float(number)
