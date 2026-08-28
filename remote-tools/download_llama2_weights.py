from pathlib import Path
import hashlib
from huggingface_hub import hf_hub_download

OUT = Path('/root/models/Llama-2-7b-hf')
REPO = 'NousResearch/Llama-2-7b-hf'
EXPECTED = {
    'model-00001-of-00002.safetensors': '4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41',
    'model-00002-of-00002.safetensors': '41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f',
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

OUT.mkdir(parents=True, exist_ok=True)
for name, expected in EXPECTED.items():
    target = OUT / name
    if target.exists() and target.stat().st_size == (9976578928 if name.startswith('model-00001') else 3500297344):
        got = sha256(target)
        if got == expected:
            print(f'OK existing {name} {got}', flush=True)
            continue
        target.unlink()
    print(f'DOWNLOAD {name}', flush=True)
    cached = hf_hub_download(repo_id=REPO, filename=name, local_dir=str(OUT), local_dir_use_symlinks=False)
    got = sha256(Path(cached))
    print(f'CHECK {name} {got}', flush=True)
    if got != expected:
        raise RuntimeError(f'hash mismatch for {name}: {got} != {expected}')
print('DOWNLOAD_COMPLETE', flush=True)
