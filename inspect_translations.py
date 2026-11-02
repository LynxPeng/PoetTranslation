"""只读检查译文开头及note标记；不修改译文，不调用API。"""
import argparse
import json
from pathlib import Path
import re


NOTE = re.compile(r'note:|Note:|NOTE:')


def empty_stats():
    return dict(records=0, checked_translations=0, invalid_translation=0,
                non_english_start=0, raw_non_english_start=0, contains_note=0,
                note_occurrences=0, note_variants={'note:': 0, 'Note:': 0, 'NOTE:': 0})


def inspect_rows(rows):
    summary, by_model, matches = empty_stats(), {}, []
    for line, row in rows:
        track = row.get('generator_track', 'unknown')
        stats = by_model.setdefault(track, empty_stats())
        text = row.get('translation')
        valid = isinstance(text, str) and bool(text.strip())
        hits = list(NOTE.finditer(text)) if valid else []
        bad_start = valid and not re.match(r'[A-Za-z]', text.lstrip())
        raw_bad_start = valid and not re.match(r'[A-Za-z]', text)
        for counts in (summary, stats):
            counts['records'] += 1
            counts['checked_translations'] += int(valid)
            counts['invalid_translation'] += int(not valid)
            counts['non_english_start'] += int(bool(bad_start))
            counts['raw_non_english_start'] += int(bool(raw_bad_start))
            counts['contains_note'] += int(bool(hits))
            counts['note_occurrences'] += len(hits)
            for hit in hits:counts['note_variants'][hit.group()] += 1
        if not valid or bad_start or raw_bad_start or hits:
            matches.append(dict(
                line=line, **{key: row.get(key) for key in ('id', 'generator_track', 'condition_id', 'poem_id', 'run_id', 'status')},
                invalid_translation=not valid, non_english_start=bool(bad_start),
                raw_non_english_start=bool(raw_bad_start), contains_note=bool(hits),
                first_character=text.lstrip()[:1] if valid else None,
                preview=text[:180] if valid else text,
                note_matches=[{'marker': hit.group(), 'offset': hit.start(),
                               'context': text[max(0, hit.start()-35):hit.end()+100]} for hit in hits]))
    return {'summary': summary, 'by_model': by_model, 'matches': matches}


def read_rows(path):
    with path.open(encoding='utf-8-sig') as file:
        for line, text in enumerate(file, 1):
            if not text.strip():continue
            try:row = json.loads(text)
            except json.JSONDecodeError as exc:raise ValueError(f'{path}:{line}: {exc.msg}') from exc
            if not isinstance(row, dict):raise ValueError(f'{path}:{line}: 每行须为JSON对象')
            yield line, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('run_config.json')))
    parser.add_argument('--input', type=Path, help='待检查的结果JSONL，默认取配置generation_output')
    parser.add_argument('--report', type=Path, default=Path(__file__).parent / 'results/translation_inspection.json')
    args = parser.parse_args()
    if args.input:
        source = args.input.resolve()
    else:
        config_path = Path(args.config).resolve()
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        source = (config_path.parent / config['generation_output']).resolve()
    if source == args.report.resolve():raise ValueError('检查报告不能覆盖输入文件')
    report = inspect_rows(read_rows(source))
    report['input'] = str(source)
    report['rules'] = {'non_english_start': '去除开头空白后，首字符不属于ASCII A-Z/a-z；引号、标点、数字、中文、重音字母均计入',
                       'raw_non_english_start': '不去除开头空白的同项统计',
                       'contains_note': '任意位置包含note:、Note:或NOTE:之一；一份译文只计一次，出现次数另计',
                       'invalid_translation': '缺失、非字符串或空白译文单独统计，不并入首字符统计'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('input', 'summary', 'by_model')}, ensure_ascii=False, indent=2))
    print('report:', args.report.resolve())


if __name__ == '__main__':
    main()
