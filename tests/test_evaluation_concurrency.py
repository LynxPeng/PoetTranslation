import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import evaluate
import generate


class EvaluationConcurrencyTests(unittest.TestCase):
    def run_fixture(self, texts, responder, output_hook=None, workers=3, retry=False):
        root = Path(__file__).resolve().parents[1]
        config, _ = generate.load_config(root / 'run_config.json')
        for key in ('poems', 'rubric', 'judge_prompt'):
            config[key] = str(root / config[key])
        poem = generate.read_json(config['poems'])['poems'][0]
        with tempfile.TemporaryDirectory() as folder:
            config['generation_output'] = str(Path(folder) / 'input.jsonl')
            config['evaluation_output'] = str(Path(folder) / 'scores.jsonl')
            for index, text in enumerate(texts):
                generate.append_jsonl(config['generation_output'], dict(id=str(index), poem_id=poem['id'],
                    source={key: poem[key] for key in ('title', 'author', 'text')}, translation=text,
                    generator_track='mimo', condition_id='G1-C00', run_id=index, status='ok'))
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            args = ['evaluate.py', '--config', str(path), '--workers', str(workers)]
            real_append = generate.append_jsonl
            def append(target, row):
                if output_hook:output_hook(row)
                real_append(target, row)
            error = None
            with contextlib.redirect_stdout(io.StringIO()), patch.object(evaluate, 'chat', side_effect=responder) as caller, \
                    patch.object(evaluate, 'append_jsonl', side_effect=append), patch('sys.argv', args):
                try:
                    evaluate.main()
                except SystemExit as exc:
                    error = exc
                if error is None:
                    if retry:
                        with patch('sys.argv', args + ['--retry-failed']), patch.object(evaluate, 'chat', return_value=self.response()) as retry_caller:
                            evaluate.main()
                        self.assertEqual(retry_caller.call_count, 1)
                    evaluate.main()  # 完成后续跑不得再次收费。
            rows = list(generate.read_jsonl(config['evaluation_output'])) if Path(config['evaluation_output']).exists() else []
            return rows, caller.call_count, error

    @staticmethod
    def response():
        scores = dict(evaluate.SCORE_LIMITS, total=100)
        return {'choices': [{'message': {'content': json.dumps(scores)}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'prompt_cache_hit_tokens': 80}}

    def test_parallel_unique_requests_warmup_dedup_and_main_thread_writes(self):
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        calls, active, maximum, writers = [], 0, 0, []
        def responder(model, body):
            nonlocal active, maximum
            text = json.loads(body['messages'][1]['content'])['translation']
            with lock:
                calls.append(text)
                active += 1
                maximum = max(maximum, active)
            if text != 'A':barrier.wait(timeout=3)
            with lock:active -= 1
            return self.response()
        rows, count, error = self.run_fixture(['D', 'B', 'A', 'B', 'C'], responder,
            lambda row: writers.append(threading.get_ident()))
        self.assertIsNone(error)
        self.assertEqual(count, 4)
        self.assertEqual(calls[0], 'A')
        self.assertEqual(maximum, 3)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(row['billable_call'] for row in rows), 4)
        self.assertEqual(set(writers), {threading.get_ident()})
        self.assertTrue(all(row['status'] == 'ok' for row in rows))

    def test_failure_preserves_inflight_and_continues(self):
        barrier = threading.Barrier(3)
        def responder(model, body):
            text = json.loads(body['messages'][1]['content'])['translation']
            if text != 'A':barrier.wait(timeout=3)
            if text == 'B':return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
            return self.response()
        rows, count, error = self.run_fixture(['A', 'B', 'C', 'D'], responder)
        self.assertIsNone(error)
        self.assertEqual(count, 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row['status'] == 'ok' for row in rows), 3)
        self.assertEqual(sum(row['status'] == 'error' for row in rows), 1)
        failed = next(row for row in rows if row['status'] == 'error')
        self.assertIn('response', failed)

    def test_first_request_failure_still_sends_others(self):
        rows, count, error = self.run_fixture(['A', 'B', 'C'], lambda *args: self.response() | {
            'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]})
        self.assertIsNone(error)
        self.assertEqual(count, 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row['status'] == 'error' for row in rows))

    def test_transport_failure_is_saved_and_dispatch_continues(self):
        def responder(model, body):
            text = json.loads(body['messages'][1]['content'])['translation']
            if text == 'B':raise OSError('connection failed')
            return self.response()
        rows, count, error = self.run_fixture(['A', 'B', 'C', 'D'], responder, workers=1)
        self.assertIsNone(error)
        self.assertEqual(count, 4)
        self.assertEqual([row['generation_id'] for row in rows], ['0', '1', '2', '3'])
        self.assertEqual(rows[1]['error'], 'connection failed')

    def test_duplicate_failed_input_is_not_retried_or_scored(self):
        def responder(model, body):
            text = json.loads(body['messages'][1]['content'])['translation']
            if text == 'B':return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
            return self.response()
        rows, count, error = self.run_fixture(['A', 'B', 'B', 'C'], responder, workers=1)
        self.assertIsNone(error)
        self.assertEqual(count, 3)
        self.assertEqual(len(rows), 4)
        failed = [row for row in rows if row['status'] == 'error']
        self.assertEqual(len(failed), 2)
        self.assertEqual(sum(row['billable_call'] for row in failed), 1)
        self.assertTrue(all('total' not in row and 'response' in row for row in failed))

    def test_explicit_retry_only_failed_unique_input_preserves_history(self):
        def responder(model, body):
            text = json.loads(body['messages'][1]['content'])['translation']
            if text == 'B':return {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}]}
            return self.response()
        rows, count, error = self.run_fixture(['A', 'B', 'B', 'C'], responder, workers=1, retry=True)
        self.assertIsNone(error)
        self.assertEqual(count, 3)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row['status'] == 'error' for row in rows), 2)
        latest = {row['id']: row for row in rows}
        self.assertEqual(len(latest), 4)
        self.assertTrue(all(row['status'] == 'ok' for row in latest.values()))
