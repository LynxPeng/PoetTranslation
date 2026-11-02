"""独立核验推荐组合的16份译文、统一评分及步骤，导出可阅读结果。"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import mean

from generate import read_json, read_jsonl
from run_combination import normalize_response, pending_searches


def check_tool_completion(step, raw, active):
    message = raw['choices'][0]['message']
    if not pending_searches(message):
        return
    if '.search.' in step:
        # 强制的单次搜索已取得真实来源；模型追加建议的搜索未执行，程序只用annotations。
        assert raw.get('usage', {}).get('web_search_usage', {}).get('tool_usage') == 1
        assert message.get('annotations'), '不能将未执行的查询当搜索资料'
        return
    base = step.split('.continue')[0]
    turn = int(step.split('.continue')[1]) if '.continue' in step else 0
    assert any(s['step_id'] == f'{base}.continue{turn + 1}' for s in active), \
        '未执行的工具请求缺少完成后的后续回答'


def summarize(config_path='run_combination.json'):
    path = Path(config_path).resolve()
    root, config = path.parent, read_json(path)
    folder = (root / config['generation_output']).parent
    poems = read_json(root / config['poems'])['poems']
    translations = list(read_jsonl(root / config['generation_output']))
    assert len(translations) == len(poems) == 16
    assert len({r['id'] for r in translations}) == 16
    assert len({r['poem_id'] for r in translations}) == 16
    by_poem = {r['poem_id']: r for r in translations}
    step_history = list(read_jsonl(root / config['realtime_steps_output']))
    steps = {(r['poem_id'], r['step_id']): r for r in step_history}
    latest_scores = {r['id']: r for r in read_jsonl(root / config['evaluation_output'])}
    protocol = read_json(root / config['rubric'])['protocol_version']
    generation_ids = {r['id'] for r in translations}
    scores = {r['generation_id']: r for r in latest_scores.values() if r['protocol_version'] == protocol
              and r['generation_id'] in generation_ids}
    assert len(scores) == 16
    totals, table, tokens, search_counts = [], [], {}, {}
    for poem in poems:
        row = by_poem[poem['id']]
        assert row['status'] == 'ok' and row['run_id'] == 1
        assert row['source'] == {k: poem[k] for k in ('title', 'author', 'text')}
        assert isinstance(row['translation'], str) and row['translation'].strip()
        score = scores[row['id']]
        assert score['status'] == 'ok' and score['poem_id'] == poem['id']
        judged = json.loads(score['request']['messages'][1]['content'])
        assert judged['source'] == row['source'] and judged['translation'] == row['translation']
        assert Decimal(str(score['total'])) == sum(Decimal(str(score[f'D{i}'])) for i in range(1, 8))
        assert score['request']['thinking'] == {'type': 'disabled'}
        active = [s for s in steps.values() if s['poem_id'] == poem['id'] and s['status'] == 'ok']
        expected = {'analysis', 'select', 'estimate', 'final'} | {
            f'candidate_{model}_{i}' for model in ('mimo', 'qwen38') for i in range(1, 3)}
        assert expected <= {s['step_id'] for s in active}
        assert all(s['step_id'].split('.')[0] in expected for s in active)
        for entry in active:
            body, raw = entry['request'], entry['response']
            model = body['model']
            if model == 'mimo-v2.6-flash':
                assert body['thinking'] == {'type': 'enabled'}
                assert 'temperature' not in body and 'top_p' not in body
                searches = raw.get('usage', {}).get('web_search_usage', {}).get('tool_usage', 0)
            else:
                assert model == 'qwen3.8-flash'
                assert body['enable_thinking'] and body['thinking_budget'] == 4096
                searches = raw.get('usage', {}).get('x_tools', {}).get('web_search', {}).get('count', 0)
                if 'output' in raw:
                    raw = normalize_response(raw)
            if entry['step_id'] == 'final':
                assert body['response_format']['type'] == 'json_schema'
                assert 'tools' not in body
                assert json.loads(raw['choices'][0]['message']['content'])['translation'].strip() == row['translation']
            elif '.search.' in entry['step_id']:
                assert body['tools'][0]['type'] == 'web_search'
                assert body['tools'][0]['max_keyword'] == body['tools'][0]['limit'] == 1
                assert body['max_completion_tokens'] <= 2048
                assert searches <= 1
            else:
                assert 'response_format' not in body
                if entry['step_id'].split('.')[0] in ('select', 'estimate') or entry['step_id'].startswith('candidate_'):
                    assert 'tools' not in body, '候选/综合/修订环节不得联网'
                elif 'tools' in body:
                    assert body['tools'][0]['type'] == 'function'
            usage = raw.get('usage') or {}
            check_tool_completion(entry['step_id'], raw, active)
            bucket = tokens.setdefault(model, {'input_tokens': 0, 'cached_tokens': 0, 'output_tokens': 0,
                                               'reasoning_tokens': 0, 'api_calls': 0})
            bucket['api_calls'] += 1
            bucket['input_tokens'] += usage.get('prompt_tokens', 0) or 0
            bucket['cached_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
            bucket['output_tokens'] += usage.get('completion_tokens', 0) or 0
            bucket['reasoning_tokens'] += (usage.get('completion_tokens_details') or
                                           usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0
            search_counts[model] = search_counts.get(model, 0) + searches
        totals.append(score['total'])
        table.append({'poem_id': poem['id'], 'title': poem['title'],
                      **{f'D{i}': score[f'D{i}'] for i in range(1, 8)}, 'total': score['total']})
    before = read_json(folder / 'original_hashes.before.json')
    after = {file: hashlib.sha256((root / file).read_bytes()).hexdigest() for file in before}
    assert before == after, '历史原始文件被修改'
    reservations = list(read_jsonl(folder / 'search_budget.jsonl')) if (folder / 'search_budget.jsonl').exists() else []
    assert len(reservations) <= 16
    assert len({r['poem_id'] for r in reservations}) == len(reservations)
    assert sum(search_counts.values()) <= 16
    summary = {'status': 'complete', 'poems': 16, 'translations_ok': 16, 'scores_ok': 16,
               'logical_generation_stages': 128, 'search_limit_total': 16,
               'search_reservations': len(reservations),
               'generation_steps_ok': sum(r['status'] == 'ok' for r in steps.values()),
               'generation_failures_current': 0,
               'score_mean': mean(totals), 'score_min': min(totals), 'score_max': max(totals),
               'tokens': tokens, 'search_usage': search_counts, 'original_hashes_unchanged': True,
               'original_hashes': after, 'score_protocol': protocol}
    (folder / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    text = ['# 推荐组合单轮实测结果', '',
            'MiMo-V2.6-Flash主导分析、综合译稿及一轮TEaR；MiMo与Qwen3.8-Flash各2候选，'
            '共4份，4例高质量示范。整批先完成译前疑问核验，搜索合计最多16次且简短返回；'
            '候选、综合及修订均不联网，候选仅供参考，可融合改写。MiMo开启思考，Qwen预算4096；DeepSeek终评不思考。', '',
            f'16首各一份最终译文，16份均评分成功。平均总分 **{mean(totals):.2f}**。', '',
            '| 作品 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | 总分 |',
            '|---|' + '---:|' * 8]
    for row in table:
        text.append('| ' + row['title'] + ' | ' + ' | '.join(f'{row[k]:.2f}' for k in [
            *(f'D{i}' for i in range(1, 8)), 'total']) + ' |')
    for poem in poems:
        text += ['', '## ' + poem['id'] + ' ' + poem['title'], '',
                 by_poem[poem['id']]['translation'].replace('\n', '  \n')]
    (folder / 'results.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='run_combination.json')
    summarize(parser.parse_args().config)
