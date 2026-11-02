"""Qwen3.8全量独立轨道：逐诗生成、统一排序评分、仅补失败，可续跑。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import generate

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'run_qwen38_all.json'
OUTPUT=ROOT/'results/translations.qwen38_all.jsonl'
SCORES=ROOT/'results/scores.qwen38_all.jsonl'
LOGS=ROOT/'results/qwen38_all_logs'


def prepare():
    config,_=generate.load_config(ROOT/'run_config.json')
    config['models']['qwen38']=config['models'].pop('qwen')
    config['models']['qwen38']['model']='qwen3.8-flash'
    config['generation_tracks']=['mimo','qwen38']
    config['generation_output']=str(OUTPUT.relative_to(ROOT))
    config['evaluation_output']=str(SCORES.relative_to(ROOT))
    config['realtime_steps_output']='results/realtime_steps.qwen38_all.jsonl'
    config['steps_output']='results/steps.qwen38_all.jsonl'
    config['batch_generation_output']='results/translations.qwen38_all.batch.jsonl'
    config['generation_workers']=256
    config['evaluation_workers']=128
    config['configuration_version']='qwen38-all54-poem-locality-v1'
    for recipe in config['experiments'].values():
        for step in recipe.get('steps',[]):
            if step.get('model')=='qwen':step['model']='qwen38'
    if CONFIG.exists() and generate.read_json(CONFIG)!=config:raise ValueError('已有配置不同，不覆盖')
    CONFIG.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    work=generate.normal_flow_work(list(generate.jobs(config,ROOT,'qwen38')))
    assert len(work)==4320 and len({r['id'] for r,_ in work})==4320
    f12=list(generate.read_jsonl(ROOT/'results/translations.f12_qwen38.jsonl'))
    prior_scores=list(generate.read_jsonl(ROOT/'results/scores.f12_qwen38.jsonl'))
    assert len(f12)==480 and all(r['status']=='ok' for r in f12)
    # 已完成F12原样复制到独立总轨道，避免重新生成和评分；源文件不改。
    if not OUTPUT.exists():generate.write_jsonl_atomic(OUTPUT,f12)
    if not SCORES.exists():generate.write_jsonl_atomic(SCORES,prior_scores)
    ids={r['id'] for r,_ in work}|{r['id'] for r in f12}
    assert len(ids)==4800
    (ROOT/'experiments/qwen38_all_manifest.json').write_text(json.dumps(sorted(ids)),encoding='utf-8')
    # 新模型已观测的关闭思考延迟，结合原Qwen相同实验的各诗平均token量估算并发。
    latency=defaultdict(list)
    for row in f12:
        if row['condition_id'] not in ('G2-C09','G2-C10'):continue
        for entry in row.get('trace',[]):
            raw=entry.get('response',{})
            seconds=sum(a.get('seconds',0) for a in raw.get('_attempts',[]))
            if seconds>0:latency[row['poem_id']].append(seconds)
    tokens=defaultdict(dict)
    for row in generate.read_jsonl(ROOT/'results/translations.json_mode_no_thinking.jsonl'):
        if row['generator_track']!='qwen' or row['condition_id'] not in config['conditions']:continue
        for entry in row.get('trace',[]):
            body=entry.get('request',{});raw=entry.get('response',{})
            if body.get('model')!='qwen-flash':continue
            usage=raw.get('usage',{})
            if usage:tokens[row['poem_id']][entry['step_record_id']]=usage.get('prompt_tokens',0)+usage.get('completion_tokens',0)
    poems=generate.read_json(ROOT/config['poems'])['poems']
    estimates=[]
    for poem in poems:
        pid=poem['id'];cost=statistics.mean(tokens[pid].values()) if tokens[pid] else 1500
        seconds=statistics.mean(latency[pid]) if latency[pid] else 3.5
        workers=min(256,max(16,math.floor(4_000_000*seconds/(60*cost))))
        estimates.append(dict(poem_id=pid,mean_tokens=round(cost),observed_latency_seconds=round(seconds,3),
                              workers=workers,estimated_tpm=round(workers*cost*60/seconds)))
    LOGS.mkdir(exist_ok=True)
    (LOGS/'concurrency_estimate.json').write_text(json.dumps(estimates,ensure_ascii=False,indent=2),encoding='utf-8')
    generate.log_event({'event':'prepared','new_tasks':4320,'reused_f12':480,'generation_concurrency_cap':256,
                        'workers_by_poem':{r['poem_id']:r['workers'] for r in estimates},'evaluation_workers':128})
    return estimates


def call(script,args,name):
    path=LOGS/f'{name}.log'
    with path.open('a',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-X','utf8',script,'--config',str(CONFIG),*args],cwd=ROOT,
                              stdout=stream,stderr=subprocess.STDOUT)
    generate.log_event({'event':'stage_finished','stage':name,'exit_code':result.returncode,'log':str(path)})
    if result.returncode not in (0,1):raise RuntimeError(f'{name}退出异常')


def state():
    translations={r['id']:r for r in generate.read_jsonl(OUTPUT)}
    scores={r['id']:r for r in generate.read_jsonl(SCORES)}
    return translations,scores


def finish():
    translations,scores=state()
    expected=set(generate.read_json(ROOT/'experiments/qwen38_all_manifest.json'))
    successful_scores={r['generation_id'] for r in scores.values() if r['status']=='ok'}
    summary={'expected':len(expected),'generation_status':dict(Counter(r['status'] for r in translations.values())),
             'score_status':dict(Counter(r['status'] for r in scores.values())),
             'missing_generation':len(expected-translations.keys()),'missing_successful_scores':len(expected-successful_scores),
             'new_conditions':54,'reused_f12':480,'timestamp':datetime.now().astimezone().isoformat()}
    (LOGS/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    groups=defaultdict(list)
    for score in scores.values():
        if score['status']=='ok':groups[score['condition_id']].append(score['total'])
    lines=['# Qwen3.8-Flash全量实验','',json.dumps(summary,ensure_ascii=False),'',
           '原54条件新增4320份，已完成F12的480份原样复用；独立总轨道4800份。普通步骤关闭思考，合作条件保留MiMo搭档。',
           '', '| 条件 | 已评分 | 平均总分 |','|---|---:|---:|']
    lines += [f'| {c} | {len(v)} | {statistics.mean(v):.3f} |' for c,v in sorted(groups.items())]
    (ROOT/'docs/qwen38_all_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    generate.log_event(summary)
    if len(translations)!=4800 or any(r['status']!='ok' for r in translations.values()) or summary['missing_successful_scores']:
        raise RuntimeError('仍有失败或缺项，见summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    estimates=prepare()
    if args.prepare_only:return
    # 每首诗内先由生产入口预热，再并发；多轮完整流程也集中在本诗。
    for estimate in estimates:
        pid=estimate['poem_id']
        generate.log_event({'event':'poem_start',**estimate})
        call('generate.py',['--model','qwen38','--workers',str(estimate['workers']),'--poems',pid],f'generation_{pid}')
    for attempt in range(1,3):
        translations,_=state()
        if not any(r['status']=='faild' for r in translations.values()):break
        call('generate.py',['--model','qwen38','--workers','128','--retry-failed'],f'generation_retry_{attempt}')
    call('evaluate.py',['--workers','128'],'evaluation')
    for attempt in range(1,3):
        _,scores=state()
        if not any(r['status']!='ok' for r in scores.values()):break
        call('evaluate.py',['--workers','128','--retry-failed'],f'evaluation_retry_{attempt}')
    finish()


if __name__=='__main__':main()
