import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from generate import load, generate


def main():
    model, tokenizer, cfg = load()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'status': 'ok', 'model': 'XSLM'})
            else:
                self.reply(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/generate':
                return self.reply(404, {'error': 'not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('request size')
                data = json.loads(self.rfile.read(length))
                prompt = data['prompt']
                tokens = data.get('tokens', cfg['tokens'])
                temperature = data.get('temperature', 0.8)
                top_k = data.get('top_k', 8)
                seed = data.get('seed', cfg['seed'])
                if not isinstance(prompt, str) or not 0 < len(prompt) <= 512:
                    raise ValueError('prompt')
                if type(tokens) is not int or not 1 <= tokens <= cfg['max_tokens']:
                    raise ValueError('tokens')
                if type(top_k) is not int or top_k < 0 or type(seed) is not int:
                    raise ValueError('top_k/seed')
                if type(temperature) not in (int, float):
                    raise ValueError('temperature')
                result = generate(model, tokenizer, prompt, tokens, temperature, top_k, seed)
                self.reply(200, {'text': result, 'model': 'XSLM'})
            except (ValueError, KeyError, TypeError):
                self.reply(400, {'error': 'invalid request'})

    address = ('127.0.0.1', cfg['port'])
    print('XSLM Runner:', address, flush=True)
    HTTPServer(address, Handler).serve_forever()


if __name__ == '__main__':
    main()
