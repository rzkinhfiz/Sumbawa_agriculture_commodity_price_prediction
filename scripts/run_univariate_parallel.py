import yaml
from pathlib import Path
import subprocess
import sys
import multiprocessing as mp

root = Path('.').resolve()
cfg = yaml.safe_load((root / 'config.yaml').read_text())
commodities = [c['series_id'] for c in cfg['commodities']]

# Decide concurrency: if CUDA available, run sequentially to avoid GPU contention
import torch
has_cuda = torch.cuda.is_available()

cmd_template = [sys.executable, '-m', 'src.train', '--epochs', '60', '--device', 'auto', '--log-csv', 'runs/univariate_metrics.csv', '--batch-size', str(cfg['model']['univariate'].get('batch_size', 64))]


def run_one(commodity):
    cmd = cmd_template + ['--commodity', commodity]
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=root)
    return proc.returncode


if __name__ == '__main__':
    if has_cuda:
        print('CUDA available — running commodities sequentially to avoid GPU contention')
        results = []
        for c in commodities:
            rc = run_one(c)
            results.append((c, rc))
    else:
        print('No CUDA — running commodities in parallel on CPU')
        with mp.Pool(min(len(commodities), mp.cpu_count())) as p:
            results_codes = p.map(run_one, commodities)
        results = list(zip(commodities, results_codes))

    print('\nRun results:')
    for c, rc in results:
        print(c, '->', 'OK' if rc == 0 else f'RC={rc}')
