import math


class Value:
    def __init__(self, data, parents=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, ((self, 1), (other, 1)))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data,
                     ((self, other.data), (other, self.data)))

    def __pow__(self, power):
        return Value(self.data ** power,
                     ((self, power * self.data ** (power - 1)),))

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    __radd__ = __add__
    __rmul__ = __mul__

    def exp(self):
        result = math.exp(self.data)
        return Value(result, ((self, result),))

    def log(self):
        return Value(math.log(self.data), ((self, 1 / self.data),))

    def relu(self):
        return Value(max(0, self.data), ((self, float(self.data > 0)),))

    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent, _ in node.parents)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            for parent, derivative in node.parents:
                parent.grad += derivative * node.grad


def softmax(values):
    peak = max(v.data for v in values)
    exps = [(v - peak).exp() for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def linear(x, weights):
    return [sum(a * b for a, b in zip(x, row)) for row in weights]


def norm(x):
    mean = sum(x) / len(x)
    variance = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) * (variance + 1e-5) ** -0.5 for v in x]


if __name__ == '__main__':
    x = Value(3)
    y = x * x + 2 * x
    y.backward()
    print('y =', y.data, 'dy/dx =', x.grad)
    assert y.data == 15 and x.grad == 8
