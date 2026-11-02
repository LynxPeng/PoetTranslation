"""准备及核验F12补充实验；网络调用仍使用generate.py与evaluate.py。"""
import argparse
import copy
import json
from pathlib import Path

import generate

ROOT = Path(__file__).resolve().parent
CONDITIONS = ['G2-C05', 'G2-C06', 'G2-C07', 'G2-C08', 'G2-C09', 'G2-C10']


def build_config(track):
    config, _ = generate.load_config(ROOT / 'run_config.json')
    source_analysis = copy.deepcopy(config['experiments']['G2-C02']['steps'][0])
    config['experiments'] = {}
    config['conditions'] = CONDITIONS[:] if track == 'qwen' else [c for c in CONDITIONS if c not in ('G2-C06', 'G2-C08')]
    config.update(design='experiments/f12_design.json', normal_flow_fixes=True,
                  configuration_version='f12-budget-supplement-v1', repeats=5,
                  realtime_steps_output='results/realtime_steps.f12.jsonl')
    base = copy.deepcopy(config['models'][track])
    for key in ('thinking_budget', 'thinking_completion_allowance'):base.pop(key, None)
    base['thinking_enabled'] = False
    config['models'] = {track: base, 'formatter': copy.deepcopy(base),
                        'deepseek': copy.deepcopy(config['models']['deepseek'])}
    if track == 'qwen':
        config['models']['medium'] = dict(base, thinking_enabled=True, thinking_budget=4096)
        config['models']['high'] = dict(base, thinking_enabled=True, thinking_budget=16384)
    else:config['models']['enabled'] = dict(base, thinking_enabled=True)
    config['models']['partner'] = copy.deepcopy(base)  # 本实验无跨模型节点，满足共用入口预检。
    config['generation_tracks'] = [track, 'partner']
    for condition in config['conditions']:
        off = condition in ('G2-C09', 'G2-C10')
        analysis = condition in ('G2-C07', 'G2-C08', 'G2-C10')
        mode = track if off else ('enabled' if track == 'mimo' else 'high' if condition in ('G2-C06', 'G2-C08') else 'medium')
        steps = []
        if analysis:
            item = copy.deepcopy(source_analysis)
            item.pop('max_tokens_by_hanzi', None)
            item.update(model=mode, max_tokens=32768)
            steps.append(item)
        draft = {'id': 'draft', 'model': mode, 'max_tokens': 32768, 'output_format': 'text',
                 'sample_slot': 'f12_translation',
                 'prompt': ('请根据对话中的完整原作和译前分析，完成整首英译。分析可能有误，始终以原作为准。只给完整英文译文正文。'
                            if analysis else '将给定完整作品译成英语诗。只给完整英文译文正文。\n待译作品数据：\n{{source}}')}
        if analysis:draft['history_from'] = 'analysis'
        steps.append(draft)
        steps.append({'id': 'translation', 'model': 'formatter', 'max_tokens': 32768,
                      'output_format': 'translation', 'temperature': 0, 'sample_slot': 'f12_json_pack',
                      'exact_copy_from': 'draft',
                      'prompt': '仅封装下面已有英文译文。逐字保留全部内容、措辞、标点和诗行换行，不润色、不修订、不增删。\n{{draft}}'})
        config['experiments'][condition] = {'steps': steps, 'factors': ['F12'] + (['F10'] if analysis else []),
            'f12': {'mode': 'off' if off else mode, 'analysis': analysis,
                    'matched_control': 'G2-C10' if analysis else 'G2-C09'}}
    return config


def prepare():
    design = {'groups': [{'id': 'G2-F12-supplement', 'conditions': [{'id': c} for c in CONDITIONS]}],
              'unsupported': {'mimo': ['G2-C06', 'G2-C08']},
              'labels': {'G2-C05': 'Qwen中4096 / MiMo开启，无分析', 'G2-C06': 'Qwen高16384，无分析',
                         'G2-C07': 'Qwen中4096 / MiMo开启，有分析', 'G2-C08': 'Qwen高16384，有分析',
                         'G2-C09': '无思考，无分析，匹配封装对照', 'G2-C10': '无思考，有分析，匹配封装对照'}}
    (ROOT / 'experiments/f12_design.json').write_text(json.dumps(design, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = {'protocol': 'f12-budget-supplement-v1', 'tasks': []}
    for track in ('mimo', 'qwen'):
        config = build_config(track)
        (ROOT / f'run_f12_{track}.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for row, _ in generate.jobs(config, ROOT, track):
            manifest['tasks'].append({k: row[k] for k in ('id', 'generator_track', 'condition_id', 'poem_id', 'run_id')})
    target = ROOT / 'experiments/f12_manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    generate.log_event({'event': 'f12_prepared', 'tasks': len(manifest['tasks']), 'manifest': str(target)})


def audit():
    from collections import Counter
    manifest = generate.read_json(ROOT / 'experiments/f12_manifest.json')
    ids = {row['id'] for row in manifest['tasks']}
    config = generate.read_json(ROOT / 'run_config.json')
    rows = {row['id']: row for row in generate.read_jsonl(ROOT / config['generation_output']) if row['id'] in ids}
    ledger = generate.load_step_cache(ROOT / 'results/realtime_steps.f12.jsonl')
    issues, usage = [], Counter()
    conditions = {}
    for row in rows.values():
        if row['status'] != 'ok':
            issues.append({'id': row['id'], 'error': row.get('error')})
            continue
        key = row['generator_track'] + '/' + row['condition_id']
        conditions[key] = conditions.get(key, 0) + 1
        calls = {item['step']: ledger[item['step_record_id']] for item in row['trace'] if 'step_record_id' in item}
        body = calls['draft']['request']
        draft = generate.response_text(calls['draft']['response'], natural_intermediate=True,
            allow_thinking=body.get('enable_thinking', body.get('thinking', {}).get('type') == 'enabled'))
        if generate.copy_text(draft) != generate.copy_text(row['translation']):issues.append({'id': row['id'], 'error': '译文与封装前正文不同'})
        for name, entry in calls.items():
            body = entry['request']
            enabled = body.get('enable_thinking', body.get('thinking', {}).get('type') == 'enabled')
            expected = row['condition_id'] not in ('G2-C09', 'G2-C10') and name != 'translation'
            if enabled != expected:issues.append({'id': row['id'], 'error': '思考开关不符', 'step': name})
    responses = {}
    for entry in generate.read_jsonl(ROOT / 'results/realtime_steps.f12.jsonl'):
        if entry['status'] == 'ok' and entry.get('response'):
            responses[entry['response'].get('id', generate.digest(entry['response']))] = entry
    for entry in responses.values():
        raw = entry.get('response', {})
        if entry['status'] != 'ok':continue
        provider = 'qwen' if entry['request']['model'].startswith('qwen') else 'mimo'
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[provider + '/' + name] += raw.get('usage', {}).get(name, 0)
        usage[provider + '/reasoning_tokens'] += (raw.get('usage', {}).get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
    result = {'expected': len(ids), 'recorded': len(rows), 'missing': len(ids - rows.keys()),
              'statuses': dict(Counter(row['status'] for row in rows.values())), 'conditions': conditions,
              'issues': issues, 'actual_api_usage': dict(usage)}
    target = ROOT / 'results/f12_audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    generate.log_event(result)


def report():
    from collections import defaultdict
    import statistics
    ids = {row['id'] for row in generate.read_json(ROOT / 'experiments/f12_manifest.json')['tasks']}
    config = generate.read_json(ROOT / 'run_config.json')
    translations = list(generate.read_jsonl(ROOT / config['generation_output']))
    subset = {row['id']: row for row in translations if row['id'] in ids}
    scores = {}
    version = generate.read_json(ROOT / config['rubric'])['protocol_version']
    for row in generate.read_jsonl(ROOT / config['evaluation_output']):
        if row.get('protocol_version') == version and row['generation_id'] in ids:
            scores[row['generation_id']] = row
    grouped = defaultdict(list)
    paired = {}
    for key, score in scores.items():
        if score['status'] != 'ok':continue
        row = subset[key]
        grouped[(row['generator_track'], row['condition_id'])].append(score['total'])
        paired[(row['generator_track'], row['condition_id'], row['poem_id'], row['run_id'])] = score['total']
    summary = []
    for (track, condition), values in sorted(grouped.items()):
        control = 'G2-C10' if condition in ('G2-C07', 'G2-C08', 'G2-C10') else 'G2-C09'
        diffs = [score - paired[(t, control, poem, run)] for (t, c, poem, run), score in paired.items()
                 if t == track and c == condition and (t, control, poem, run) in paired]
        summary.append({'track': track, 'condition': condition, 'n': len(values), 'mean': statistics.mean(values),
                        'control': control, 'paired_n': len(diffs), 'mean_difference': statistics.mean(diffs) if diffs else None})
    integrity = generate.read_json(ROOT / 'results/f12_baseline_integrity.json')
    baseline = [row for row in translations if row['id'] not in ids]
    result = {'expected': len(ids), 'generated': len(subset), 'scored_ok': sum(row['status'] == 'ok' for row in scores.values()),
              'scored_failed': sum(row['status'] != 'ok' for row in scores.values()), 'conditions': summary,
              'original_8640_unchanged': len(baseline) == integrity['count'] and generate.digest(baseline) == integrity['hash']}
    (ROOT / 'results/f12_scores_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# F12思考预算补充实测', '', f'新增任务{len(ids)}份；已生成{len(subset)}份；评分成功{result["scored_ok"]}份，评分失败{result["scored_failed"]}份。', '',
             'Qwen：中档4096、高档16384原生思考token上限；MiMo：仅支持开/关。所有条件16首×5次。G2-C09/C10为同封装流程的无思考匹配对照。', '',
             '| 轨道 | 条件 | 有效评分 | 平均总分 | 匹配对照 | 配对均差 |', '|---|---|---:|---:|---|---:|']
    for item in summary:
        lines.append(f'| {item["track"]} | {item["condition"]} | {item["n"]} | {item["mean"]:.3f} | {item["control"]} | {item["mean_difference"]:.3f} |')
    lines += ['', '每诗均为5次且等权；均差为同诗同run配对差，不代表统计显著或人工文学结论。预算是最大值，模型未必耗尽，因此中/高预算不保证实际思考长度不同。', '',
              '生成步骤与分析使用指定思考模式，末轮关闭思考只封装JSON，程序核验措辞、标点和分行一致（只忽略行末空格与CRLF）。原生reasoning_content留在台账，不传后续对话。MiMo思考模式原生不接受temperature/top_p，开关比较同时受此采样限制。', '',
              '两模型都计入可观察的HTTP响应usage，见results/f12_audit.json；包括封装、失败和修复调用，不等同于平台账单。原8640份译文逐字段未变：' + str(result['original_8640_unchanged']) + '。', '',
              '配置：run_f12_mimo.json、run_f12_qwen.json；条件标签：experiments/f12_design.json；新增ID：experiments/f12_manifest.json；评分按id取最后一条。']
    (ROOT / 'docs/f12_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    generate.log_event(result)


def recover():
    """仅重解析已返回的完整台账，不调用API，不改成功记录。"""
    import shutil
    from datetime import datetime
    config = generate.read_json(ROOT / 'run_config.json')
    output = ROOT / config['generation_output']
    rows = list(generate.read_jsonl(output))
    index = {row['id']: i for i, row in enumerate(rows)}
    cache = generate.load_step_cache(ROOT / 'results/realtime_steps.f12.jsonl')
    recovered = []
    for track in ('mimo', 'qwen'):
        current, _ = generate.load_config(ROOT / f'run_f12_{track}.json')
        models = {key: generate.model_config(current, key) for key in current['models']}
        work = generate.normal_flow_work(list(generate.jobs(current, ROOT, track)))
        for row, recipe in work:
            position = index.get(row['id'])
            if position is None or rows[position]['status'] == 'ok':continue
            try:ready, finished = generate.batch_wave([(row, recipe)], models[track], cache, models)
            except (ValueError, KeyError, TypeError):continue
            if not ready and finished:
                result = finished[0]
                result['timestamp'] = datetime.now().astimezone().isoformat(timespec='milliseconds')
                result['offline_recovery'] = {'prior_error': rows[position].get('error'), 'reason': 'reparse existing responses; no API call'}
                rows[position] = result
                recovered.append(row['id'])
    if recovered:
        backup = ROOT / 'results/recovery' / ('f12_offline_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jsonl')
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output, backup)
        generate.write_jsonl_atomic(output, rows)
    generate.log_event({'event': 'f12_offline_recovery', 'recovered': len(recovered), 'ids': recovered})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'audit', 'recover', 'report'])
    args = parser.parse_args()
    if args.action == 'recover':
        with generate.generation_lock(ROOT / 'results/generation'):recover()
    else:{'prepare': prepare, 'audit': audit, 'report': report}[args.action]()
