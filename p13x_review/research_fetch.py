import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent


def fetch(item):
    name, url = item
    try:
        response = requests.get(url, timeout=45)
        folder = ROOT / 'sources'
        folder.mkdir(exist_ok=True)
        body = response.text
        (folder / (name + '.html')).write_text(body, encoding='utf-8')
        soup = BeautifulSoup(body, 'html.parser')
        text = soup.get_text('\n', strip=True)
        (folder / (name + '.txt')).write_text(text, encoding='utf-8')
        return {'name': name, 'url': url, 'status': response.status_code,
                'sha256': hashlib.sha256(response.content).hexdigest(), 'text': text}
    except Exception as error:
        return {'name': name, 'url': url, 'error': str(error)}


if __name__ == '__main__':
    items = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    results = list(concurrent.futures.ThreadPoolExecutor(max_workers=5).map(fetch, items))
    (ROOT / (Path(sys.argv[1]).stem + '_results.json')).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != 'text'}, ensure_ascii=False))
        text = result.get('text', '')
        if '/search/' in result['url']:
            soup = BeautifulSoup((ROOT / 'sources' / (result['name'] + '.html')).read_text(encoding='utf-8'), 'html.parser')
            entries = soup.select('li.arxiv-result')
            print('Results on page:', len(entries))
            for entry in entries:
                print(entry.select_one('.list-title').get_text(' ', strip=True),
                      entry.select_one('.title').get_text(' ', strip=True))
            if not entries:
                print(text[:1000])
        else:
            print(text[:int(sys.argv[2]) if len(sys.argv) > 2 else 2000])
