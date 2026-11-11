"""Check model files, numerical results and executable source before installing PyTorch."""
from hashlib import sha256
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
def main():
    files=json.loads((ROOT/'checksums.json').read_text(encoding='utf8'))
    for relative,expected in files.items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit('Missing file: '+relative)
        if path.stat().st_size!=expected['bytes'] or sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise SystemExit('Changed file: '+relative)
    print(f'Verified {len(files)} files')
if __name__=='__main__':
    main()
