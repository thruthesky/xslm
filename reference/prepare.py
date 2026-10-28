from pathlib import Path


lines = [line.strip() for line in Path('data/raw.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
lines = list(dict.fromkeys(lines))
assert len(lines) >= 5, 'write at least five different Q/A lines'
assert all(' | A:' in line and line.startswith('Q:') for line in lines)
train = [line for i, line in enumerate(lines) if i % 5 != 0]
val = [line for i, line in enumerate(lines) if i % 5 == 0]
for name, rows in [('train', train), ('val', val)]:
    text = '\n'.join(row.replace(' | A:', '\nA:') for row in rows) + '\n'
    Path('data/' + name + '.txt').write_text(text, encoding='utf-8')
    print(name, len(rows))
