"""为用户授权的Qwen3.8-Flash F12单独轨道准备配置，不调用API。"""
import copy
import json
from pathlib import Path
import generate

ROOT=Path(__file__).resolve().parent


def main():
    config=copy.deepcopy(generate.read_json(ROOT/'run_f12_qwen.json'))
    config['models']['qwen38']=config['models'].pop('qwen')
    for key,model in config['models'].items():
        if key!='deepseek':model['model']='qwen3.8-flash'
    config['generation_tracks']=['qwen38','partner']
    config['configuration_version']='f12-qwen3.8-flash-isolated-v1'
    config['generation_output']='results/translations.f12_qwen38.jsonl'
    config['realtime_steps_output']='results/realtime_steps.f12_qwen38.jsonl'
    config['evaluation_output']='results/scores.f12_qwen38.jsonl'
    config['steps_output']='results/steps.f12_qwen38.jsonl'
    config['batch_generation_output']='results/translations.f12_qwen38.batch.jsonl'
    config['evaluation_workers']=32
    for recipe in config['experiments'].values():
        for step in recipe['steps']:
            if step.get('model')=='qwen':step['model']='qwen38'
    path=ROOT/'run_f12_qwen38.json'
    if path.exists() and generate.read_json(path)!=config:
        raise ValueError('已有配置不同，禁止覆盖运行中的配置')
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    work=generate.normal_flow_work(list(generate.jobs(config,ROOT,'qwen38')))
    assert len(work)==480 and len({row['id'] for row,_ in work})==480
    preflight=[row['id'] for row,_ in work if row['poem_id']=='T01' and row['run_id']==1
               and row['condition_id'] in ('G2-C05','G2-C06','G2-C09')]
    assert len(preflight)==3
    (ROOT/'experiments/f12_qwen38_manifest.json').write_text(json.dumps(
        [{'id':r['id'],'condition':r['condition_id'],'poem':r['poem_id'],'run':r['run_id']} for r,_ in work],
        ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'experiments/f12_qwen38_preflight.json').write_text(json.dumps(preflight),encoding='utf-8')
    print(json.dumps({'tasks':480,'preflight':3,'model':'qwen3.8-flash','track':'qwen38'}))


if __name__=='__main__':main()
