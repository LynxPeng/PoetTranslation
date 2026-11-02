from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import generate


class TagConcurrencyTests(unittest.TestCase):
    def test_translation_tags_preserve_text_and_notes_rules(self):
        step = {'output_format': 'translation', 'translation_format': 'tag'}
        self.assertEqual(generate.decode_step(step, '<translation>Moon & frost\nHome.</translation>', {}),
                         ('Moon & frost\nHome.', []))
        self.assertEqual(generate.decode_step(step, 'translation>\nMoon\n</translation>', {}), ('Moon', []))
        for text in ('Moon', '<translation></translation>',
                     '<translation>Moon</translation><translation>Home</translation>'):
            with self.assertRaises(ValueError):generate.decode_step(step, text, {})
        for text in ('Intro<translation>Moon</translation>After',
                     '<translations>\n<translation>Moon</translation>\n</translations>',
                     '```xml\n<translation>Moon</translation>\n```',
                     '<translation>Moon</translation><notes>extra</notes>'):
            self.assertEqual(generate.decode_step(step, text, {}), ('Moon', []))
        note_step = dict(step, notes_allowed=True, notes_required=True)
        self.assertEqual(generate.decode_step(note_step, '<translation>Moon</translation>\n<notes>依据</notes>', {}), ('Moon', ['依据']))
        with self.assertRaises(ValueError):generate.decode_step(note_step, '<translation>Moon</translation>', {})

    def test_prompt_preamble_is_before_source_and_no_translation_json(self):
        expected = '<要求>将翻译后的完整纯文本放入一对<translation>和</translation>标签中，保留诗行换行。</要求>\n\n'
        self.assertEqual(generate.TRANSLATION_PREAMBLE, expected)
        recipe = {'steps': [{'id': 'draft', 'prompt': '将原作译成英语诗。只输出JSON对象，translation为完整英译，notes=[]。原作：{{source}}', 'output_format': 'translation'}]}
        prepared = generate.source_first_recipe(generate.translation_tag_recipe(recipe))
        step = prepared['steps'][0]
        body = generate.stage_body(step, {'source': '原诗'}, {'provider': 'mimo', 'model': 'mimo-v2.6-flash'}, 1)
        text = body['messages'][0]['content']
        self.assertTrue(text.startswith(generate.TRANSLATION_PREAMBLE))
        self.assertLess(text.index('纯文本'), text.index('<原诗>'))
        self.assertNotIn('只输出JSON', text)
        self.assertNotIn('<translation>', recipe['steps'][0]['prompt'])
        internal = generate.translation_tag_recipe({'steps': [{'id': 'analysis', 'prompt': '分析原作'}]})
        self.assertEqual(internal['steps'][0]['output_preamble'], expected)

    def test_generation_ignores_surrounding_thought_tags_but_keeps_raw(self):
        model = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'thinking_enabled': True}
        row = {'id': 'job', 'poem_id': 'T01', 'run_id': 1, 'source': {'text': '月'}}
        recipe = {'steps': [{'id': 'draft', 'prompt': '{{source}}', 'output_format': 'translation', 'translation_format': 'tag'}]}
        ready, _ = generate.batch_wave([(row, recipe)], model, {})
        raw = {'choices': [{'message': {'content': '<think>outside</think><translations><translation>Moon</translation></translations>'}, 'finish_reason': 'stop'}]}
        cache = {key: dict(entry, status='ok', response=raw) for key, entry in ready.items()}
        _, finished = generate.batch_wave([(row, recipe)], model, cache)
        self.assertEqual(finished[0]['translation'], 'Moon')
        self.assertIn('<think>outside</think>', next(iter(cache.values()))['response']['choices'][0]['message']['content'])

    def test_cache_usage_reports_real_numbers_and_missing_values(self):
        self.assertEqual(generate.cache_usage({'usage': {'prompt_tokens': 200, 'prompt_tokens_details': {'cached_tokens': 150}}}),
                         {'prompt_tokens': 200, 'cached_tokens': 150, 'cache_hit_rate': '75.00%'})
        self.assertEqual(generate.cache_usage({'usage': {'prompt_tokens': 200}})['cached_tokens'], None)
        self.assertEqual(generate.cache_usage({'usage': {'prompt_tokens': 200}})['cache_hit_rate'], None)
        self.assertIsNone(generate.cache_usage({'usage': {'prompt_tokens_details': 'malformed'}})['cached_tokens'])

    def test_cache_session_rate_is_token_weighted_and_reuse_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = generate.RealtimeSteps({}, Path(folder) / 'steps.jsonl', log_api=True)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                for prompt, cached in ((100, 90), (900, 90)):
                    entry = {'endpoint': 'local', 'request': {'model': 'model'}, 'status': 'ok',
                             'response': {'usage': {'prompt_tokens': prompt, 'prompt_tokens_details': {'cached_tokens': cached}}}}
                    coordinator.emit(entry, {}, True)
                coordinator.emit(entry, {}, False)
            logs = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(logs[1]['session_cache_hit_rate'], '18.00%')
            self.assertEqual(logs[1]['session_cache_reported_calls'], 2)
            self.assertEqual(logs[2]['event'], 'step_reused')
            self.assertIsNone(logs[2]['cached_tokens'])

    @staticmethod
    def work():
        single = {'steps': [{'id': 'draft', 'prompt': '{{source}}'}]}
        return [({'id': f'{poem}-{run}', 'poem_id': poem, 'run_id': run}, single)
                for poem in ('T01', 'T02') for run in range(1, 5)]

    def test_poem_warmup_and_barrier_with_bounded_parallelism(self):
        lock = threading.Lock()
        completed, active, peak = [], 0, 0
        overlap = threading.Barrier(3, timeout=5)
        def execute(job):
            nonlocal active, peak
            row, _ = job
            with lock:
                if row['run_id'] != 1:self.assertIn(row['poem_id'] + '-1', completed)
                if row['poem_id'] == 'T02':self.assertTrue(all(f'T01-{n}' in completed for n in range(1, 5)))
                active += 1;peak = max(peak, active)
            if row['run_id'] != 1:overlap.wait()
            with lock:active -= 1;completed.append(row['id'])
            return True
        self.assertTrue(generate.run_realtime_work(self.work(), execute, 3))
        self.assertEqual(peak, 3)
        self.assertEqual(len(completed), 8)

    def test_failed_warmup_does_not_dispatch_more(self):
        calls = []
        def execute(job):calls.append(job[0]['id']);return False
        self.assertFalse(generate.run_realtime_work(self.work(), execute, 3))
        self.assertEqual(calls, ['T01-1'])

    def test_parallel_failure_stops_next_poem(self):
        calls = []
        def execute(job):
            calls.append(job[0]['id'])
            return job[0]['run_id'] != 2
        self.assertFalse(generate.run_realtime_work(self.work(), execute, 3))
        self.assertTrue(all(value.startswith('T01-') for value in calls))
        self.assertLessEqual(len(calls), 4)

    def test_current_cli_tags_cache_metrics_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        config, _ = generate.load_config(root / 'run_config.json')
        config['models']['mimo']['thinking_enabled'] = True  # 原始思考保存兼容用例。
        for key in ('poems', 'design'):config[key] = str(root / config[key])
        config.update(conditions=['G1-C00', 'G5-C01', 'G2-C03'], repeats=2)
        config['final_json_output'] = False
        config.pop('materials')
        def response(model, body):
            text = '<translation>Moon & frost\nHome.</translation>'
            if '先完成完整英译' in body['messages'][0]['content']:text += '\n原文依据的简短说明'
            return {'choices': [{'message': {'content': text, 'reasoning_content': 'mock reasoning'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 200, 'prompt_tokens_details': {'cached_tokens': 128},
                              'completion_tokens_details': {'reasoning_tokens': 1}}}
        with tempfile.TemporaryDirectory() as folder:
            config['generation_output'] = str(Path(folder) / 'translations.jsonl')
            config['realtime_steps_output'] = str(Path(folder) / 'steps.jsonl')
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch('sys.argv', ['generate.py', '--config', str(path), '--poems', 'T01', 'T02']), \
                    patch.object(generate, 'chat', side_effect=response) as caller:
                generate.main()
                generate.main()
                self.assertEqual(caller.call_count, 12)
            logs = [json.loads(line) for line in stdout.getvalue().splitlines()]
            api = [item for item in logs if item.get('event') == 'api_response']
            self.assertEqual(len(api), 12)
            self.assertTrue(all(item['cached_tokens'] == 128 and item['cache_hit_rate'] == '64.00%' for item in api))
            self.assertEqual(api[-1]['session_cache_hit_rate'], '64.00%')
            rows = list(generate.read_jsonl(config['generation_output']))
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row['translation'] == 'Moon & frost\nHome.' for row in rows))
            self.assertTrue(all('notes' not in row for row in rows))
            records = list(generate.read_jsonl(config['realtime_steps_output']))
            self.assertEqual(sum(r['status'] == 'ok' for r in records), 12)
            self.assertTrue(all(r['response']['choices'][0]['message']['reasoning_content'] == 'mock reasoning' for r in records if r['status'] == 'ok'))

    def test_shared_inflight_request_only_calls_once(self):
        model = {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'base_url': 'http://local/v1'}
        recipe = {'steps': [{'id': 'draft', 'prompt': '{{source}}'}]}
        ready = threading.Barrier(3, timeout=5)
        raw = {'choices': [{'message': {'content': 'Moon'}, 'finish_reason': 'stop'}],
               'usage': {'prompt_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 64}}}
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / 'steps.jsonl'
            coordinator = generate.RealtimeSteps({}, ledger)
            def run(_):
                ready.wait()
                return generate.run_recipe(recipe, {'source': '{}'}, model, seed=1, coordinator=coordinator)[0]
            with patch.object(generate, 'chat', return_value=raw) as caller, ThreadPoolExecutor(max_workers=3) as pool:
                self.assertEqual(list(pool.map(run, range(3))), ['Moon'] * 3)
                self.assertEqual(caller.call_count, 1)
            records = list(generate.read_jsonl(ledger))
            self.assertEqual([r['status'] for r in records], ['running', 'ok'])


if __name__ == '__main__':unittest.main()
