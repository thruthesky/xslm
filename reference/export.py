import argparse
import json
import shutil
import struct
from pathlib import Path

SOURCES = ['engine.py', 'model.py', 'tokenizer.py', 'generate.py',
           'runner.py', 'xchat.py', 'export.py']


def write_safetensors(path, weights):
    header, body = {'__metadata__': {'format': 'xslm'}}, b''
    for name, matrix in weights.items():
        rows, cols = len(matrix), len(matrix[0])
        data = struct.pack('<%df' % (rows * cols),
                           *[v for row in matrix for v in row])
        header[name] = {'dtype': 'F32', 'shape': [rows, cols],
                        'data_offsets': [len(body), len(body) + len(data)]}
        body += data
    head = json.dumps(header).encode('utf-8')
    head += b' ' * (-len(head) % 8)
    path.write_bytes(struct.pack('<Q', len(head)) + head + body)


def read_safetensors(path):
    raw = path.read_bytes()
    size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    body, weights = raw[8 + size:], {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        assert info['dtype'] == 'F32', 'only float32'
        rows, cols = info['shape']
        start, end = info['data_offsets']
        flat = struct.unpack('<%df' % (rows * cols), body[start:end])
        weights[name] = [list(flat[r * cols:(r + 1) * cols])
                         for r in range(rows)]
    return weights


def export(folder):
    saved = json.loads(Path('checkpoints/model.json').read_text())
    folder.mkdir(exist_ok=True)
    write_safetensors(folder / 'model.safetensors', saved['weights'])
    config = {'model_type': 'xslm', 'vocab_size': len(saved['tokenizer']['vocab']),
              'step': saved['step'], 'settings': saved['settings']}
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    (folder / 'tokenizer.json').write_text(
        json.dumps(saved['tokenizer'], ensure_ascii=False), encoding='utf-8')
    for name in SOURCES:
        shutil.copy(name, folder / name)
    loaded = read_safetensors(folder / 'model.safetensors')
    gap = max(abs(a - b) for name in loaded
              for row, old in zip(loaded[name], saved['weights'][name])
              for a, b in zip(row, old))
    count = sum(len(row) for matrix in loaded.values() for row in matrix)
    print('parameters:', count, 'bytes:', (folder / 'model.safetensors').stat().st_size)
    print('largest float32 gap:', '%.1e' % gap)
    print('exported:', folder)


def restore(folder):
    config = json.loads((folder / 'config.json').read_text())
    state = {'settings': config['settings'],
             'tokenizer': json.loads((folder / 'tokenizer.json').read_text(encoding='utf-8')),
             'weights': read_safetensors(folder / 'model.safetensors')}
    Path('checkpoints').mkdir(exist_ok=True)
    Path('checkpoints/model.json').write_text(
        json.dumps(state, ensure_ascii=False), encoding='utf-8')
    print('restored: checkpoints/model.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--to', default='hub')
    parser.add_argument('--restore', metavar='FOLDER')
    args = parser.parse_args()
    if args.restore:
        restore(Path(args.restore))
    else:
        export(Path(args.to))
