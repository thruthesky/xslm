import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError


def ask(question):
    cfg = json.loads(Path('settings.json').read_text())
    body = {'prompt': 'Q: ' + question + '\nA: ', 'tokens': cfg['tokens']}
    request = Request('http://127.0.0.1:' + str(cfg['port']) + '/generate',
                      data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)['text']


if __name__ == '__main__':
    print('XChat -> Runner -> XSLM | /exit')
    while True:
        try:
            question = input('You> ').strip()
            if question == '/exit':
                break
            if question:
                print('XSLM>', ask(question))
        except (EOFError, KeyboardInterrupt):
            break
        except (URLError, TimeoutError) as error:
            print('Runner:', error)
