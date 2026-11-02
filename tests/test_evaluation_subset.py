import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
import generate


class EvaluationSubsetTests(unittest.TestCase):
    def test_sorted_subset_deduplicates_logs_cache_and_resume_does_not_expand(self):
        root = Path(__file__).resolve().parents[1]
        config, _ = generate.load_config(root / 'run_config.json')
        for key in ('poems', 'rubric', 'judge_prompt'):config[key] = str(root / config[key])
        poems = generate.read_json(config['poems'])['poems']
        with tempfile.TemporaryDirectory() as folder:
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['evaluation_output'] = str(Path(folder) / 'unused_scores.jsonl')
            output = Path(folder) / 'subset_scores.jsonl'
            for i, text in enumerate(('Z', 'A', 'A', 'B', 'C', 'other poem')):
                poem = poems[0 if i < 5 else 1]
                generate.append_jsonl(config['generation_output'], dict(id=str(i), poem_id=poem['id'],
                    source={k: poem[k] for k in ('title', 'author', 'text')}, translation=text,
                    generator_track='mimo', condition_id='G1-C00', run_id=i, status='ok'))
            path = Path(folder) / 'config.json';path.write_text(json.dumps(config), encoding='utf-8')
            args = ['evaluate.py', '--config', str(path), '--poems', 'T01', '--limit', '3', '--output', str(output)]
            response = {'choices': [{'message': {'content': '{"D1":30,"D2":10,"D3":15,"D4":15,"D5":10,"D6":10,"D7":10,"total":100}'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'prompt_cache_hit_tokens': 80}}
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.object(evaluate, 'chat', return_value=response) as caller, patch('sys.argv', args):
                evaluate.main()
                evaluate.main()
            self.assertEqual(caller.call_count, 2)
            rows = list(generate.read_jsonl(output))
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['generation_id'] for r in rows}, {'1', '2', '3'})
            self.assertEqual(sum(r['billable_call'] for r in rows), 2)
            events = [json.loads(line) for line in stdout.getvalue().splitlines()]
            api = [e for e in events if e.get('event') == 'api_response']
            self.assertEqual(len(api), 2)
            self.assertTrue(all(e['cached_tokens'] == 80 and e['cache_hit_rate'] == '80.00%' for e in api))
            self.assertEqual(api[-1]['session_cache_hit_rate'], '80.00%')
            self.assertTrue(all('timestamp' in e for e in api))
            self.assertTrue(any(e.get('event') == 'step_reused' and e['cached_tokens'] is None for e in events))

