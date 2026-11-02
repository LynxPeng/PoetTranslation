"""独立DeepSeek评分入口：统一规则，仅提供冻结原文与匿名译文。"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal
import json
import sys

from generate import (RealtimeSteps, append_jsonl, chat, digest, log_event,
                      load_config, model_config, read_json, read_jsonl,
                      request_body, response_text)


SCORE_LIMITS = dict(zip(('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'), (30, 10, 15, 15, 10, 10, 10)))


def parse_score(text):
    data = json.loads(text)
    limits = dict(SCORE_LIMITS, total=100)
    if not isinstance(data, dict) or set(data) != set(limits):
        raise ValueError('评分JSON必须恰好包含D1—D7及total')
    numbers = {}
    for key, limit in limits.items():
        value = data[key]
        if type(value) not in (int, float):raise ValueError(key + '必须是数字')
        number = Decimal(str(value))
        if not number.is_finite() or not 0 <= number <= limit or number * 10 != (number * 10).to_integral_value():
            raise ValueError(f'{key}须在0—{limit}之间，最多1位小数')
        numbers[key] = number
    if sum(numbers[key] for key in SCORE_LIMITS) != numbers['total']:
        raise ValueError('total必须等于D1—D7的小分之和')
    return {key: float(value) for key, value in numbers.items()}


def evaluation_order(item):
    row, body = item[:2]
    return body['messages'][0]['content'], row['poem_id'], row['translation'], digest(row['id'])


def judge_messages(poem, rubric, translation, instructions):
    common = {"source": {k: poem[k] for k in ("title", "author", "text")}}
    rules = {k: rubric[k] for k in ("task", "dimensions", "internal_anchors", "total_formula")}
    return [{"role": "system", "content": instructions + "\n" + json.dumps(rules, ensure_ascii=False, sort_keys=True)},
            {"role": "user", "content": json.dumps(dict(common, translation=translation), ensure_ascii=False)}]


def score_request(model, body):
    """线程只请求和校验；日志、缓存及JSONL统一由主线程处理。"""
    raw = None
    try:
        raw = chat(model, body)
        return raw, parse_score(response_text(raw)), None
    except Exception as exc:
        return raw, None, exc


def parallel_scores(groups, model, workers):
    """先完成首个请求，再有限并发派发；单项失败不阻塞其余任务。"""
    pending = iter(groups)
    first = next(pending, None)
    if first is None:return
    result = score_request(model, first[0][1])
    yield first, result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight = {}
        while True:
            while len(inflight) < workers:
                group = next(pending, None)
                if group is None:break
                inflight[pool.submit(score_request, model, group[0][1])] = group
            if not inflight:break
            completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
            results = [(inflight.pop(future), future.result()) for future in completed]
            for group, result in results:
                yield group, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="run_config.json")
    parser.add_argument("--input", help="生成结果JSONL；默认取配置generation_output")
    parser.add_argument('--output', help='评分输出JSONL；默认取配置evaluation_output')
    parser.add_argument('--poems', nargs='+', help='只评指定诗歌ID，如T01')
    parser.add_argument('--limit', type=int, help='排序后固定选前N条结果（含重复映射），续跑不扩大范围')
    parser.add_argument('--workers', type=int, help='并发评分请求数，默认8；1为串行')
    parser.add_argument('--retry-failed', action='store_true', help='仅重跑已有失败评分；保留历史，追加最新结果')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:raise ValueError('--limit必须是正整数')
    config, root = load_config(args.config)
    workers = args.workers if args.workers is not None else config.get('evaluation_workers', 8)
    if type(workers) is not int or workers < 1:raise ValueError('--workers必须是正整数')
    model = model_config(config, config.get("judge", "deepseek"))
    if model.get("provider") != "deepseek" or model.get("model") != "deepseek-flash":
        raise ValueError("终评必须使用DeepSeek Flash")
    poems = {p["id"]: p for p in read_json(root / config["poems"])["poems"]}
    if args.poems and set(args.poems) - poems.keys():raise ValueError('未知诗歌ID')
    rubric = read_json(root / config.get("rubric", "evaluation/rubric.json"))
    if {d['id']: d['weight'] for d in rubric['dimensions']} != SCORE_LIMITS:
        raise ValueError('量表须使用统一七维权重30/10/15/15/10/10/10')
    prompt = (root / config.get("judge_prompt", "evaluation/judge_prompt.md")).read_text(encoding="utf-8")
    instructions = prompt.split("## A. 高优先级指令正文", 1)[1].split("## B.", 1)[0].strip()
    input_path = root / (args.input or config.get("generation_output", "results/translations.jsonl"))
    output = root / (args.output or config.get("evaluation_output", "results/scores.jsonl"))
    if input_path.resolve() == output.resolve():
        raise ValueError("评分输出不能覆盖生成输入")
    previous = list(read_jsonl(output)) if output.exists() else []
    latest = {v['id']: v for v in previous}
    done = set(latest)
    score_cache = {v["evaluation_key"]: v for v in previous if "evaluation_key" in v}
    if args.retry_failed:
        score_cache = {key: value for key, value in score_cache.items() if value['status'] == 'ok'}
    planned = []
    seen = set()
    system_prompts = set()
    for row in read_jsonl(input_path):
        if row["id"] in seen:
            raise ValueError("输入存在重复生成ID")
        seen.add(row["id"])
        if args.poems and row['poem_id'] not in args.poems:continue
        if row.get("status") != "ok":
            continue  # 生成失败是操作失败，不伪造译文评分。
        poem = poems[row["poem_id"]]
        expected = {k: poem[k] for k in ("title", "author", "text")}
        if row["source"] != expected or not row["translation"].strip():
            raise ValueError("源文与冻结数据不符，或译文为空")
        messages = judge_messages(poem, rubric, row["translation"], instructions)
        system_prompts.add(messages[0]['content'])
        body = request_body(model, messages, 512, temperature=0)
        body['response_format'] = {'type': 'json_object'}
        evaluation_key = digest([body, model["base_url"]])
        key = digest([row["id"], evaluation_key])
        planned.append((row, body, key, evaluation_key))
    if len(system_prompts) > 1:raise ValueError('所有评分请求必须使用相同的系统提示词')
    planned.sort(key=evaluation_order)
    if args.limit is not None:planned = planned[:args.limit]
    selected = len(planned)
    selected_ids = {item[2] for item in planned}
    outcomes = {row['id']: row['status'] for row in previous if row['id'] in selected_ids}
    if args.retry_failed:
        planned = [item for item in planned if item[2] in latest and latest[item[2]]['status'] != 'ok']
    else:
        planned = [item for item in planned if item[2] not in done]
    unique_api_requests = len({item[3] for item in planned if item[3] not in score_cache})
    plan = {'selected': selected, 'pending': len(planned), 'unique_api_requests': unique_api_requests,
            'local_reuse_records': len(planned) - unique_api_requests, 'output': str(output),
            'poems': args.poems, 'limit': args.limit, 'system_prompt_count': len(system_prompts),
            'system_prompt_hash': digest(next(iter(system_prompts))) if system_prompts else None,
            'workers': workers, 'warmup_first_request': True,
            'retry_failed': args.retry_failed,
            'order': 'system_prompt_then_poem_then_translation_lexicographic_then_anonymous_hash'}
    if args.dry_run:
        log_event(plan)
        return
    log_event(dict(plan, event='evaluation_plan'))
    cache_logger = RealtimeSteps({}, output, log_api=True)
    def save_result(item, raw=None, scores=None, error=None, cached=None):
        row, body, key, evaluation_key = item
        result = {"id": key, "generation_id": row["id"], "poem_id": row["poem_id"],
                  "generator_track": row["generator_track"], "condition_id": row["condition_id"],
                  "run_id": row["run_id"], "request_hash": digest(body), "request": body,
                  "evaluation_key": evaluation_key, 'system_prompt_hash': digest(body['messages'][0]['content']),
                  'protocol_version': rubric['protocol_version']}
        event_entry = {'request': body, 'endpoint': model['base_url'], 'step_id': 'evaluation', 'status': 'ok'}
        if cached is not None:
            if cached['status'] == 'ok':
                scores = {name: cached[name] for name in (*SCORE_LIMITS, 'total')}
                result.update(scores, status='ok', score=scores['total'])
            else:
                error = cached.get('error', '相同评分输入已有失败，未自动重评')
                result.update(status='error', error=str(error), blocked_by=cached['id'])
                if 'response' in cached:result['response'] = cached['response']
                event_entry.update(status='error', error=str(error))
            result.update(reused_from=cached['id'], billable_call=False)
            cache_logger.emit(event_entry, row, False)
        else:
            if raw is not None:result['response'] = raw
            cache_logger.emit(dict(event_entry, response=raw or {},
                                   status='error' if error is not None else 'ok',
                                   **({'error': str(error)} if error is not None else {})), row, True)
            if error is None:
                result.update(scores, status='ok', score=scores['total'], billable_call=True)
            else:
                result.update(status='error', error=str(error), attempts=getattr(error, 'attempts', []), billable_call=True)
                log_event({'event': 'evaluation_failed', 'generation_id': row['id'], 'poem_id': row['poem_id'], 'error': str(error)})
        append_jsonl(output, result)
        outcomes[key] = result['status']
        if error is None:print(json.dumps(scores, ensure_ascii=False), flush=True)
        return result

    # 在派发前统一分组，避免相同译文在多个线程中重复收费。
    groups = {}
    for item in planned:
        groups.setdefault(item[3], []).append(item)
    new_groups = []
    for evaluation_key, group in groups.items():
        if evaluation_key in score_cache:
            for item in group:save_result(item, cached=score_cache[evaluation_key])
        else:new_groups.append(group)
    for group, (raw, scores, error) in parallel_scores(new_groups, model, workers):
        result = save_result(group[0], raw, scores, error)
        score_cache[group[0][3]] = result
        for item in group[1:]:save_result(item, cached=result)
    log_event({'event': 'evaluation_summary', 'selected': selected,
               'ok': sum(status == 'ok' for status in outcomes.values()),
               'failed': sum(status != 'ok' for status in outcomes.values()),
               'pending': selected - len(outcomes), 'output': str(output)})


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
