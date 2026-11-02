"""多轮流程并发：流程间重叠，流程内保持顺序及独立历史。"""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import generate


class MultiturnConcurrencyTests(unittest.TestCase):
    def test_flows_overlap_keep_history_and_respect_worker_limit(self):
        recipe = generate.conversation_recipe({'steps': [
            {'id': 'draft', 'prompt': '{{source}}', 'output_format': 'translation'},
            {'id': 'translation', 'prompt': 'Revise {{draft}}',
             'history_from': 'draft', 'output_format': 'translation'},
        ]}, final_json=True)
        work = [({'id': str(i), 'poem_id': 'T01', 'run_id': i}, recipe) for i in range(6)]
        barrier = threading.Barrier(3, timeout=2)
        lock = threading.Lock()
        active = peak = 0
        results = {}

        def chat(model, body):
            messages = body['messages']
            if len(messages) == 1:
                token = messages[0]['content']
                barrier.wait()
                content = 'draft-' + token
            else:
                self.assertEqual(len(messages), 4)
                token = messages[1]['content']
                self.assertEqual(messages[2]['content'], 'draft-' + token)
                self.assertEqual(messages[3]['content'], 'Revise draft-' + token)
                content = '{"translation":"final-' + token + '"}'
            return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}

        with tempfile.TemporaryDirectory() as folder:
            coordinator = generate.RealtimeSteps({}, Path(folder) / 'steps.jsonl')

            def execute(job):
                nonlocal active, peak
                row, flow = job
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    result, _ = generate.run_recipe(flow, {'source': row['id']},
                        {'provider': 'mimo', 'model': 'mimo-v2.6-flash', 'base_url': 'http://local'},
                        seed=row['run_id'], coordinator=coordinator)
                    with lock:results[row['id']] = result
                finally:
                    with lock:active -= 1
                return True

            with patch.object(generate, 'chat', side_effect=chat) as caller:
                self.assertTrue(generate.run_realtime_work(work, execute, 3))
                self.assertEqual(caller.call_count, 12)
            self.assertEqual(peak, 3)
            self.assertEqual(results, {str(i): 'final-' + str(i) for i in range(6)})

    def test_one_worker_preserves_submission_order(self):
        recipe = {'steps': [{'id': 'draft'}, {'id': 'translation'}]}
        work = [({'id': str(i)}, recipe) for i in range(4)]
        calls = []
        self.assertTrue(generate.run_realtime_work(work, lambda job: calls.append(job[0]['id']) or True, 1))
        self.assertEqual(calls, ['0', '1', '2', '3'])


if __name__ == '__main__':
    unittest.main()
