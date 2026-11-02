"""将隔离运行的Qwen补测结果并回主数据；保留成功记录及合并前备份。"""
from datetime import datetime
from pathlib import Path
import shutil

import generate

ROOT = Path(__file__).resolve().parent


def merge_qwen():
    config = generate.read_json(ROOT / 'run_config.json')
    output = ROOT / config['generation_output']
    rows = list(generate.read_jsonl(output))
    index = {row['id']: i for i, row in enumerate(rows)}
    before = len(rows)
    backup = ROOT / 'results/recovery' / ('f12_merge_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    shutil.copy2(output, backup / output.name)
    for row in generate.read_jsonl(ROOT / 'results/translations.f12_qwen.jsonl'):
        position = index.get(row['id'])
        if position is None:
            index[row['id']] = len(rows)
            rows.append(row)
        elif rows[position]['status'] != 'ok' and row['status'] == 'ok':rows[position] = row
    generate.write_jsonl_atomic(output, rows)
    ledger = ROOT / 'results/realtime_steps.f12.jsonl'
    seen = {generate.digest(row) for row in generate.read_jsonl(ledger)}
    for row in generate.read_jsonl(ROOT / 'results/realtime_steps.f12_qwen.jsonl'):
        identity = generate.digest(row)
        if identity not in seen:
            generate.append_jsonl(ledger, row)
            seen.add(identity)
    generate.log_event({'event': 'f12_qwen_merged', 'new_rows': len(rows) - before, 'backup': str(backup)})


if __name__ == '__main__':
    with generate.generation_lock(ROOT / 'results/generation'):merge_qwen()
